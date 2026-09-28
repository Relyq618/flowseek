# This file includes code from SEA-RAFT (https://github.com/princeton-vl/SEA-RAFT)
# Copyright (c) 2024, Princeton Vision & Learning Lab
# Licensed under the BSD 3-Clause License

import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append('core')
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from config.parser import parse_args

import datasets
from flowseek import *
from tqdm import tqdm
from utils.utils import resize_data, load_ckpt

def expand_checkpoint_weights(state_dict, num_layers):
    """
    Convert original FlowSeek checkpoint keys to
    Multi-Context / Multi-Head FlowSeek keys if needed.
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
        num_layers=getattr(model, "num_layers", 4),
    )

    missing_keys, unexpected_keys = model.load_state_dict(
        state_dict,
        strict=False,
    )

    print("Loaded checkpoint:", ckpt_path)
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)

def forward_flow(args, model, image1, image2):
    output = model(image1, image2, iters=args.iters, test_mode=True)
    flow_final = output['flow'][-1]
    info_final = output['info'][-1]
    return flow_final, info_final


def resize_multilayer_flow(flow, scale_factor):
    """
    Resize flow tensor.

    Supports:
        [B, 2, H, W]
        [B, L, 2, H, W]
    """
    if flow.dim() == 4:
        flow_down = F.interpolate(
            flow,
            scale_factor=scale_factor,
            mode="bilinear",
            align_corners=False,
        ) * scale_factor
        return flow_down

    if flow.dim() == 5:
        b, l, c, h, w = flow.shape

        flow_flat = flow.reshape(
            b * l,
            c,
            h,
            w,
        )

        flow_flat = F.interpolate(
            flow_flat,
            scale_factor=scale_factor,
            mode="bilinear",
            align_corners=False,
        ) * scale_factor

        _, _, h2, w2 = flow_flat.shape

        flow_down = flow_flat.reshape(
            b,
            l,
            c,
            h2,
            w2,
        )

        return flow_down

    raise ValueError(f"Unsupported flow shape: {flow.shape}")


def resize_multilayer_info(info, scale_factor):
    """
    Resize info tensor.

    Supports:
        [B, C, H, W]
        [B, L, C, H, W]
    """
    if info.dim() == 4:
        info_down = F.interpolate(
            info,
            scale_factor=scale_factor,
            mode="area",
        )
        return info_down

    if info.dim() == 5:
        b, l, c, h, w = info.shape

        info_flat = info.reshape(
            b * l,
            c,
            h,
            w,
        )

        info_flat = F.interpolate(
            info_flat,
            scale_factor=scale_factor,
            mode="area",
        )

        _, _, h2, w2 = info_flat.shape

        info_down = info_flat.reshape(
            b,
            l,
            c,
            h2,
            w2,
        )

        return info_down

    raise ValueError(f"Unsupported info shape: {info.shape}")


def calc_flow(args, model, image1, image2):
    img1 = F.interpolate(
        image1,
        scale_factor=2 ** args.scale,
        mode="bilinear",
        align_corners=False,
    )

    img2 = F.interpolate(
        image2,
        scale_factor=2 ** args.scale,
        mode="bilinear",
        align_corners=False,
    )

    flow, info = forward_flow(
        args,
        model,
        img1,
        img2,
    )

    down_scale = 0.5 ** args.scale

    flow_down = resize_multilayer_flow(
        flow,
        down_scale,
    )

    info_down = resize_multilayer_info(
        info,
        down_scale,
    )

    return flow_down, info_down


@torch.no_grad()
def validate_sintel(args, model):
    """ Peform validation using the Sintel (train) split """
    for dstype in ['clean', 'final']:
        val_dataset = datasets.MpiSintel(split='training', dstype=dstype, root=args.paths['sintel'])
        val_loader = data.DataLoader(val_dataset, batch_size=4, 
            pin_memory=False, shuffle=False, num_workers=16, drop_last=False)
        epe_list = np.array([], dtype=np.float32)
        px1_list = np.array([], dtype=np.float32)
        px3_list = np.array([], dtype=np.float32)
        px5_list = np.array([], dtype=np.float32)
        for i_batch, data_blob in enumerate(tqdm(val_loader)):
            image1, image2, flow_gt, valid = [x.cuda(non_blocking=True) for x in data_blob]
            flow, info = calc_flow(args, model, image1, image2)
            epe = torch.sum((flow - flow_gt)**2, dim=1).sqrt()
            px1 = (epe < 1.0).float().mean(dim=[1, 2]).cpu().numpy()
            px3 = (epe < 3.0).float().mean(dim=[1, 2]).cpu().numpy()
            px5 = (epe < 5.0).float().mean(dim=[1, 2]).cpu().numpy()
            epe = epe.mean(dim=[1, 2]).cpu().numpy()
            epe_list = np.append(epe_list, epe)
            px1_list = np.append(px1_list, px1)
            px3_list = np.append(px3_list, px3)
            px5_list = np.append(px5_list, px5)

        epe = np.mean(epe_list)
        px1 = np.mean(px1_list)
        px3 = np.mean(px3_list)
        px5 = np.mean(px5_list)
        # print("Validation %s EPE: %.2f, 1px: %.2f"%(dstype,epe,100 * (1 - px1)))
        print("Validation %s EPE: %.2f"%(dstype,epe,))

@torch.no_grad()
def validate_kitti(args, model):
    """ Peform validation using the KITTI-2015 (train) split """
    val_dataset = datasets.KITTI(split='training', root=args.paths['kitti'])
    val_loader = data.DataLoader(val_dataset, batch_size=1, 
        pin_memory=False, shuffle=False, num_workers=16, drop_last=False)
    epe_list = np.array([], dtype=np.float32)
    num_valid_pixels = 0
    out_valid_pixels = 0
    for i_batch, data_blob in enumerate(tqdm(val_loader)):
        image1, image2, flow_gt, valid_gt = [x.cuda(non_blocking=True) for x in data_blob]
        flow, info = calc_flow(args, model, image1, image2)
        epe = torch.sum((flow - flow_gt)**2, dim=1).sqrt()
        mag = torch.sum(flow_gt**2, dim=1).sqrt()
        val = valid_gt >= 0.5
        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        for b in range(out.shape[0]):
            epe_list = np.append(epe_list, epe[b][val[b]].mean().cpu().numpy())
            out_valid_pixels += out[b][val[b]].sum().cpu().numpy()
            num_valid_pixels += val[b].sum().cpu().numpy()
    
    epe = np.mean(epe_list)
    f1 = 100 * out_valid_pixels / num_valid_pixels
    print("Validation KITTI: %.2f, %.1f" % (epe, f1))
    return {'kitti-epe': epe, 'kitti-f1': f1}

@torch.no_grad()
def validate_spring(args, model):
    """ Peform validation using the Spring (val) split """
    val_dataset = datasets.SpringFlowDataset(split='train', root=args.paths['spring'])
    val_loader = data.DataLoader(val_dataset, batch_size=1, 
        pin_memory=False, shuffle=False, num_workers=8, drop_last=False)
    
    epe_list = np.array([], dtype=np.float32)
    px1_list = np.array([], dtype=np.float32)
    px3_list = np.array([], dtype=np.float32)
    px5_list = np.array([], dtype=np.float32)
    for i_batch, data_blob in enumerate(tqdm(val_loader)):
        image1, image2, flow_gt, valid = [x.cuda(non_blocking=True) for x in data_blob]
        flow, info = calc_flow(args, model, image1, image2)
        epe = torch.sum((flow - flow_gt)**2, dim=1).sqrt()
        px1 = (epe < 1.0).float().mean(dim=[1, 2]).cpu().numpy()
        px3 = (epe < 3.0).float().mean(dim=[1, 2]).cpu().numpy()
        px5 = (epe < 5.0).float().mean(dim=[1, 2]).cpu().numpy()
        epe = epe.mean(dim=[1, 2]).cpu().numpy()
        epe_list = np.append(epe_list, epe)
        px1_list = np.append(px1_list, px1)
        px3_list = np.append(px3_list, px3)
        px5_list = np.append(px5_list, px5)

    epe = np.mean(epe_list)
    px1 = np.mean(px1_list)
    px3 = np.mean(px3_list)
    px5 = np.mean(px5_list)

    print("Validation Spring EPE: %.3f, 1px: %.3f"%(epe,100 * (1 - px1)))

def validate_layeredflow_first(args, model):
    """ Peform validation using the LayeredFlow (val) split """
    def datapoint_in_subset(mat, layer, subset):
        def in_list(x, l):
            return l is None or x in l
        assert type(subset) == tuple and len(subset) == 2
        return in_list(mat, subset[0]) and in_list(layer, subset[1])
        
    model.eval()
    val_dataset = datasets.LayeredFlow(downsample=8, split='val', root=args.paths['layeredflow'])

    subsets = [
        (None, (0,)), # first layer
        ((1,), (0,)), # first layer, material transparent
        ((2,), (0,)), # first layer, material reflective
        ((0,), (0,)), # first layer, material diffuse
    ]

    bad_n = [1, 3, 5]
    results = {}
    
    for subset in subsets:
        results[subset] = {}
        results[subset]['epe'] = []
        for n in bad_n:
            results[subset][str(n) + 'px'] = []

    for val_id in tqdm(range(len(val_dataset))):
        image1, image2, coords, flow_gts, materials, layers = val_dataset[val_id]
        image1, image2 = image1[None].cuda(), image2[None].cuda()
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        output = model(image1, image2, test_mode=True, demo=True)
        flow_final = output['flow'][-1]
        flow = padder.unpad(flow_final).cpu()[0]
        
        error_list = {}
        for subset in subsets:
            error_list[subset] = []
        
        for i in range(len(coords)):
            (x, y), mat, lay = coords[i], materials[i], layers[i]

            flow_pd = flow[:, x, y]
            flow_gt = torch.tensor(flow_gts[i])
            error = torch.sum((flow_pd - flow_gt)**2, dim=0).sqrt().item()

            for subset in subsets:
                if datapoint_in_subset(mat, lay, subset):
                    error_list[subset].append(error)

        for subset in subsets:
            if len(error_list[subset]) == 0:
                continue
            error_list[subset] = np.array(error_list[subset])
            results[subset]['epe'].append(np.mean(error_list[subset]))
            for n in bad_n:
                results[subset][str(n) + 'px'].extend(error_list[subset] < n)

    for subset in subsets:
        print(f"Validation LayeredFlow {subset}:")
        for key in results[subset]:
            results[subset][key] = np.mean(results[subset][key])
            if key != 'epe':
                results[subset][key] = 100 - 100 * results[subset][key]
            print(f"{key}: %.2f"%results[subset][key])

    return results
@torch.no_grad()
def validate_layeredflow_multilayer(args, model):
    """
    Validate Multi-Layer FlowSeek on LayeredFlow val split.

    Reports:
        - index-aligned EPE / Bad-1 / Bad-3 / Bad-5
        - oracle min-layer EPE / Bad-1 / Bad-3 / Bad-5
        - per-layer metrics
        - per-material metrics
    """
    model.eval()

    downsample = getattr(args, "layeredflow_downsample", 4)

    val_dataset = datasets.LayeredFlow(
        downsample=downsample,
        split="val",
        root=args.paths["layeredflow"],
    )

    print("LayeredFlow val length:", len(val_dataset))
    print("LayeredFlow downsample:", downsample)

    material_names = {
        0: "Diffuse",
        1: "Transparent",
        2: "Reflective",
    }

    def init_bucket():
        return {
            "epe_index": [],
            "epe_oracle": [],
        }

    results_all = init_bucket()

    results_by_layer = {
        0: init_bucket(),
        1: init_bucket(),
        2: init_bucket(),
        3: init_bucket(),
    }

    results_by_material = {
        0: init_bucket(),
        1: init_bucket(),
        2: init_bucket(),
    }

    bad_thresholds = [1.0, 3.0, 5.0]

    for val_id in tqdm(range(len(val_dataset))):
        image1, image2, coords, flow_gts, materials, layers = val_dataset[val_id]

        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(
            image1.shape,
            mode="kitti",
        )

        image1_pad, image2_pad = padder.pad(
            image1,
            image2,
        )

        output = model(
            image1_pad,
            image2_pad,
            test_mode=True,
            demo=True,
        )

        flow_final = output["flow"][-1]
        flow_final = padder.unpad(flow_final)

        flow = flow_final.cpu()[0]

        # Expected:
        #   multi-layer: [L, 2, H, W]
        #   single-layer: [2, H, W]
        if flow.dim() == 3:
            flow = flow[None]

        num_pred_layers = flow.shape[0]
        _, _, h, w = flow.shape

        for i in range(len(coords)):
            x, y = coords[i]
            mat = int(materials[i])
            lay = int(layers[i])

            if lay < 0 or lay >= num_pred_layers:
                continue

            if x < 0 or x >= w or y < 0 or y >= h:
                continue

            # datasets.LayeredFlow currently stores flow_gts as:
            #   (dy, dx)
            #
            # FlowSeek prediction convention:
            #   channel 0 = dx
            #   channel 1 = dy
            dy, dx = flow_gts[i]

            flow_gt = torch.tensor(
                [dx, dy],
                dtype=torch.float32,
            )

            pred_index = flow[lay, :, y, x]

            epe_index = torch.sum(
                (pred_index - flow_gt) ** 2,
                dim=0,
            ).sqrt().item()

            pred_all = flow[:, :, y, x]

            epe_all = torch.sum(
                (pred_all - flow_gt[None]) ** 2,
                dim=1,
            ).sqrt()

            epe_oracle = epe_all.min().item()

            results_all["epe_index"].append(epe_index)
            results_all["epe_oracle"].append(epe_oracle)

            if lay in results_by_layer:
                results_by_layer[lay]["epe_index"].append(epe_index)
                results_by_layer[lay]["epe_oracle"].append(epe_oracle)

            if mat in results_by_material:
                results_by_material[mat]["epe_index"].append(epe_index)
                results_by_material[mat]["epe_oracle"].append(epe_oracle)

    def summarize(name, values):
        values = np.array(values, dtype=np.float32)

        if values.size == 0:
            print(f"{name}: count=0")
            return

        print(f"{name}:")
        print(f"  count: {values.size}")
        print(f"  EPE:   {values.mean():.4f}")

        for tau in bad_thresholds:
            bad = 100.0 * np.mean(values > tau)
            print(f"  Bad-{int(tau)}: {bad:.2f}%")

    print("\n==============================")
    print("LayeredFlow Multi-Layer Result")
    print("==============================")

    summarize(
        "All / Index-aligned",
        results_all["epe_index"],
    )

    summarize(
        "All / Oracle min-layer",
        results_all["epe_oracle"],
    )

    print("\n--- Per GT Layer: Index-aligned ---")
    for lay in sorted(results_by_layer.keys()):
        summarize(
            f"Layer {lay}",
            results_by_layer[lay]["epe_index"],
        )

    print("\n--- Per GT Layer: Oracle min-layer ---")
    for lay in sorted(results_by_layer.keys()):
        summarize(
            f"Layer {lay}",
            results_by_layer[lay]["epe_oracle"],
        )

    print("\n--- Per Material: Index-aligned ---")
    for mat in sorted(results_by_material.keys()):
        summarize(
            material_names.get(mat, f"Material {mat}"),
            results_by_material[mat]["epe_index"],
        )

    print("\n--- Per Material: Oracle min-layer ---")
    for mat in sorted(results_by_material.keys()):
        summarize(
            material_names.get(mat, f"Material {mat}"),
            results_by_material[mat]["epe_oracle"],
        )
        
def eval(args):
    args.gpus = [0]
    model = FlowSeek(args)

    if args.model is not None:
        load_ckpt_multicontext(
            model,
            args.model,
        )
    model = model.cuda()
    model.eval()
    with torch.no_grad():
        if args.dataset == 'spring':
            validate_spring(args, model)
        elif args.dataset == 'sintel':
            validate_sintel(args, model)
        elif args.dataset == 'kitti':
            validate_kitti(args, model)
        elif args.dataset == 'layeredflow_first':
            validate_layeredflow_first(args, model)

        elif args.dataset == 'layeredflow':
            validate_layeredflow_multilayer(args, model)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--model', help='checkpoint path', type=str)
    parser.add_argument('--scale', help='input scale', type=int, default=0)
    parser.add_argument('--dataset', help='dataset type', type=str, required=True)
    parser.add_argument(
    "--layeredflow_downsample",
    help="downsample factor for LayeredFlow validation",
    type=int,
    default=4,
    )
    args = parse_args(parser)
    eval(args)

if __name__ == '__main__':
    main()

