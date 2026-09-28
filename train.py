# This file includes code from SEA-RAFT (https://github.com/princeton-vl/SEA-RAFT)
# Copyright (c) 2024, Princeton Vision & Learning Lab
# Licensed under the BSD 3-Clause License

import sys
sys.path.append('core')

import argparse
import numpy as np
import random

from config.parser import parse_args
from flowseek import FlowSeek

import torch
import torch.optim as optim

from datasets import fetch_dataloader
from utils.utils import load_ckpt
from loss import sequence_loss
import tqdm
import os

os.system("export KMP_INIT_AT_FORK=FALSE")



def expand_checkpoint_weights(state_dict, num_layers):
    """
    Convert original FlowSeek checkpoint keys to
    Multi-Context / Multi-Head FlowSeek keys.

    Original:
        cnet.*
        flow_head.*

    Converted:
        cnets.0.*, cnets.1.*, ...
        flow_heads.0.*, flow_heads.1.*, ...
    """
    expanded_state_dict = {}

    for key, value in state_dict.items():
        clean_key = key

        if clean_key.startswith("module."):
            clean_key = clean_key[len("module."):]

        if clean_key.startswith("flow_head."):
            suffix = clean_key[len("flow_head."):]

            for layer_idx in range(num_layers):
                new_key = f"flow_heads.{layer_idx}.{suffix}"
                expanded_state_dict[new_key] = value.clone()

        elif clean_key.startswith("cnet."):
            suffix = clean_key[len("cnet."):]

            for layer_idx in range(num_layers):
                new_key = f"cnets.{layer_idx}.{suffix}"
                expanded_state_dict[new_key] = value.clone()

        else:
            expanded_state_dict[clean_key] = value

    return expanded_state_dict


def load_ckpt_multicontext(model, ckpt_path):
    """
    Load original FlowSeek checkpoint into Multi-Context FlowSeek.
    """
    checkpoint = torch.load(
        ckpt_path,
        map_location="cpu",
    )

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    state_dict = expand_checkpoint_weights(
        state_dict,
        num_layers=model.num_layers,
    )

    missing_keys, unexpected_keys = model.load_state_dict(
        state_dict,
        strict=False,
    )

    print("Loaded checkpoint:", ckpt_path)
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)


def fetch_optimizer(args, model):
    """Create the optimizer and learning rate scheduler."""

    trainable_params = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    optimizer = optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.wdecay,
        eps=args.epsilon,
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        args.lr,
        args.num_steps + 100,
        pct_start=0.05,
        cycle_momentum=False,
        anneal_strategy='linear',
    )

    return optimizer, scheduler

def set_frozen_modules_eval(model):
    """
    Keep frozen modules in eval mode during fine-tuning.
    """
    model.dav2.eval()
    model.merge_head.eval()
    model.bnet.eval()

    if hasattr(model, "fnet"):
        model.fnet.eval()

    if hasattr(model, "update_block"):
        model.update_block.eval()

    if hasattr(model, "upsample_weight"):
        model.upsample_weight.eval()
        

def train(args, rank=0, world_size=1, use_ddp=False):
    """ Full training loop """
    device_id = rank
    model = FlowSeek(args).to(device_id)

    
    if args.restore_ckpt is not None:
        load_ckpt_multicontext(
            model,
            args.restore_ckpt,
        )

        print(f"restore ckpt from {args.restore_ckpt}")

  
    if getattr(args, "finetune_mode", None) == "cnets_flow_heads_only":
        set_finetune_trainable_modules(model)


    model = torch.nn.DataParallel(model)
    model.cuda()

    model.train()

    if getattr(args, "finetune_mode", None) == "cnets_flow_heads_only":
        set_frozen_modules_eval(model.module)

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    with open('%s/command.txt'%args.savedir, 'w') as f:
        f.write(' '.join(sys.argv))
        f.write('\n\n')
        f.write(str(args))
        f.write('\n\n')

    train_loader = fetch_dataloader(args, rank=rank, world_size=world_size, use_ddp=False)
    optimizer, scheduler = fetch_optimizer(args, model)
    total_steps = 0
    VAL_FREQ = 100
    epoch = 0
    should_keep_training = True

    while should_keep_training:
        epoch += 1
        for i_batch, data_blob in enumerate(tqdm.tqdm(train_loader)):
            optimizer.zero_grad()
            image1, image2, flow, valid = [x.to(f'cuda:{model.device_ids[0]}') for x in data_blob] 
            
            output = model(
                image1,
                image2,
                flow_gt=flow,
                iters=args.iters,
            )

            loss = sequence_loss(
                output,
                flow,
                valid,
                args.gamma,
            )

            if total_steps == 0:
                print("image1:", image1.shape)
                print("image2:", image2.shape)
                print("flow_gt:", flow.shape)
                print("valid:", valid.shape)
                print("output final:", output["final"].shape)

            if total_steps % 10 == 0:
                with torch.no_grad():
                    pred = output["final"]

                    # pred: [B, L, 2, H, W]
                    # flow: [B, L, 2, H, W]
                    epe = torch.sum(
                        (pred - flow) ** 2,
                        dim=2,
                    ).sqrt()

                    valid_mask = valid >= 0.5

                    if valid_mask.sum() > 0:
                        epe_valid = epe[valid_mask].mean()
                    else:
                        epe_valid = torch.tensor(
                            0.0,
                            device=epe.device,
                        )

                print(
                    f"step={total_steps:06d} "
                    f"loss={loss.item():.6f} "
                    f"epe_valid={epe_valid.item():.6f}"
                )

                
                for layer_idx in range(flow.shape[1]):
                    layer_valid = valid[:, layer_idx] >= 0.5

                    if layer_valid.sum() > 0:
                        layer_epe = epe[:, layer_idx][layer_valid].mean()
                        layer_valid_ratio = layer_valid.float().mean()
                    else:
                        layer_epe = torch.tensor(
                            0.0,
                            device=epe.device,
                        )
                        layer_valid_ratio = torch.tensor(
                            0.0,
                            device=epe.device,
                        )

                    print(
                        f"  layer={layer_idx} "
                        f"epe={layer_epe.item():.6f} "
                        f"valid_ratio={layer_valid_ratio.item():.6f}"
                    )


            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip) 
            optimizer.step()
            scheduler.step()

            if total_steps % VAL_FREQ == VAL_FREQ - 1 and rank == 0:
                PATH = '%s/%d_%s.pth' % (args.savedir, total_steps+1, args.name)
                torch.save(model.module.state_dict(), PATH)
            
            if total_steps > args.num_steps:
                should_keep_training = False
                break
            
            total_steps += 1

    PATH = '%s/%s.pth' % (args.savedir, args.name)
    if rank == 0:
        torch.save(model.module.state_dict(), PATH)

    return PATH

def main(rank, world_size, args, use_ddp):

    train(args, rank=rank, world_size=world_size, use_ddp=use_ddp)






def set_finetune_trainable_modules(model):
    """
    Fine-tune only layer-specific context networks and flow heads.
    """
    for param in model.parameters():
        param.requires_grad = False

    for param in model.cnets.parameters():
        param.requires_grad = True

    for param in model.flow_heads.parameters():
        param.requires_grad = True

    trainable_params = 0
    total_params = 0

    for param in model.parameters():
        numel = param.numel()
        total_params += numel

        if param.requires_grad:
            trainable_params += numel

    print(
        "Trainable parameters:",
        trainable_params,
        "/",
        total_params,
    )



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--restore_ckpt', help='restore previews weights', default=None)
    parser.add_argument(
    '--model',
    dest='restore_ckpt',
    help='alias of --restore_ckpt',
    default=None,
    )

    parser.add_argument(
    '--savedir',
    help='directory to save checkpoints and logs',
    type=str,
    default='checkpoints/debug',
    )
    parser.add_argument('--seed', help='set random seed', type=float, default=0)
    args = parse_args(parser)

    # setting random seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    main(0, 1, args, False)
    print("Done!")