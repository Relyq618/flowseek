# This file includes code from SEA-RAFT (https://github.com/princeton-vl/SEA-RAFT)
# Copyright (c) 2024, Princeton Vision & Learning Lab
# Licensed under the BSD 3-Clause License

import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append('core')
import argparse
import os
import cv2
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from config.parser import parse_args

import datasets
from flowseek import *
from utils.flow_viz import flow_to_image
from utils.utils import load_ckpt

def expand_flow_head_weights(state_dict, num_layers):
    """
    Convert original single-head FlowSeek checkpoint keys to
    multi-head FlowSeek keys.
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

        else:
            expanded_state_dict[clean_key] = value

    return expanded_state_dict


def load_ckpt_multihead(model, ckpt_path):
    """
    Load checkpoint for Multi-Head FlowSeek.

    This supports both:
        - original FlowSeek checkpoints with flow_head.*
        - new Multi-Head FlowSeek checkpoints with flow_heads.*
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

    # Remove "module." prefix if the checkpoint was saved from DataParallel/DDP.
    new_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        new_state_dict[key] = value

    new_state_dict = expand_flow_head_weights(
        new_state_dict,
        num_layers=model.num_layers,
    )

    missing_keys, unexpected_keys = model.load_state_dict(
        new_state_dict,
        strict=False,
    )

    print("Loaded checkpoint:", ckpt_path)
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)

def forward_flow(args, model, image1, image2):
    output = model(
        image1,
        image2,
        iters=args.iters,
        test_mode=True,
    )

    flow_final = output["flow"][-1]
    info_final = output["info"][-1]

    print("flow_final shape:", flow_final.shape)
    print("info_final shape:", info_final.shape)

    return flow_final, info_final

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

    # flow:
    #   [B, L, 2, H, W]
    B, L, C, H, W = flow.shape

    flow = flow.reshape(
        B * L,
        C,
        H,
        W,
    )

    flow_down = F.interpolate(
        flow,
        scale_factor=0.5 ** args.scale,
        mode="bilinear",
        align_corners=False,
    ) * (0.5 ** args.scale)

    _, _, H_down, W_down = flow_down.shape

    flow_down = flow_down.reshape(
        B,
        L,
        C,
        H_down,
        W_down,
    )

    return flow_down
def resolve_layeredflow_index(dataset, args):
    """
    Resolve LayeredFlow sample index.

    If args.scene is None:
        args.id is interpreted as a global dataset index.

    If args.scene is specified:
        args.id is interpreted as an index inside that scene.
        Example:
            --scene 6 --id 0
        means:
            the 0-th sample from scene '6'.
    """
    if args.scene is None:
        selected_index = args.id
        scene_name, scene_sample_id = dataset.data_list[selected_index]

        print(
            "Using global LayeredFlow index:",
            selected_index,
            "scene:",
            scene_name,
            "sample:",
            scene_sample_id,
        )

        return selected_index, scene_name, scene_sample_id

    scene = str(args.scene)

    if not hasattr(dataset, "data_list"):
        raise AttributeError(
            "datasets.LayeredFlow does not have data_list. "
            "Please check core/datasets.py."
        )

    matched_items = []

    for global_idx, item in enumerate(dataset.data_list):
        scene_name, scene_sample_id = item

        if str(scene_name) == scene:
            matched_items.append(
                (
                    global_idx,
                    scene_name,
                    scene_sample_id,
                )
            )

    if len(matched_items) == 0:
        available_scenes = sorted(
            list(
                set(
                    str(item[0])
                    for item in dataset.data_list
                )
            )
        )

        raise ValueError(
            f"Scene '{scene}' was not found in LayeredFlow data_list. "
            f"Available scenes include: {available_scenes[:30]}"
        )

    if args.id >= len(matched_items):
        raise IndexError(
            f"--id {args.id} is out of range for scene '{scene}'. "
            f"This scene has {len(matched_items)} samples."
        )

    selected_index, scene_name, scene_sample_id = matched_items[args.id]

    print(
        "Using LayeredFlow scene:",
        scene_name,
        "scene sample:",
        scene_sample_id,
        "global index:",
        selected_index,
    )

    return selected_index, scene_name, scene_sample_id

@torch.no_grad()
def demo_data(name, args, model, image1, image2, flow_gt, val=None):
    path = f"demo_qualitatives/{name}/"
    os.system(f"mkdir -p {path}")

    cv2.imwrite(
        f"{path}image1.jpg",
        cv2.cvtColor(
            image1[0].permute(1, 2, 0).cpu().numpy(),
            cv2.COLOR_RGB2BGR,
        ),
    )

    cv2.imwrite(
        f"{path}image2.jpg",
        cv2.cvtColor(
            image2[0].permute(1, 2, 0).cpu().numpy(),
            cv2.COLOR_RGB2BGR,
        ),
    )

    flow = calc_flow(
        args,
        model,
        image1,
        image2,
    )

    # flow:
    #   [B, L, 2, H, W]
    print("multi-layer flow shape:", flow.shape)

    num_layers = flow.shape[1]
    layer_idx = min(args.layer_idx, num_layers - 1)

    # selected_flow:
    #   [B, 2, H, W]
    selected_flow = flow[:, layer_idx]

    flow_vis = flow_to_image(
        selected_flow[0].permute(1, 2, 0).cpu().numpy(),
        convert_to_bgr=True,
    )

    cv2.imwrite(
        f"{path}flow_final_layer{layer_idx}.jpg",
        flow_vis,
    )

    # Save all predicted layers.
    if args.save_all_layers:
        for idx in range(num_layers):
            layer_flow = flow[:, idx]

            layer_flow_vis = flow_to_image(
                layer_flow[0].permute(1, 2, 0).cpu().numpy(),
                convert_to_bgr=True,
            )

            cv2.imwrite(
                f"{path}flow_final_layer{idx}.jpg",
                layer_flow_vis,
            )

    # If flow_gt is single-layer, evaluate selected layer only.
    if flow_gt is not None:
        if flow_gt.ndim == 5:
            gt_num_layers = flow_gt.shape[1]
            gt_layer_idx = min(layer_idx, gt_num_layers - 1)
            flow_gt_eval = flow_gt[:, gt_layer_idx]
        else:
            flow_gt_eval = flow_gt

        if flow_gt_eval.shape[-2:] != selected_flow.shape[-2:]:
            raise ValueError(
                f"Shape mismatch: flow_gt {flow_gt_eval.shape}, "
                f"predicted flow {selected_flow.shape}"
            )

        flow_gt_vis = flow_to_image(
            flow_gt_eval[0].permute(1, 2, 0).cpu().numpy(),
            convert_to_bgr=True,
        )

        cv2.imwrite(
            f"{path}flow_gt.jpg",
            flow_gt_vis,
        )

        diff = flow_gt_eval - selected_flow

        diff_vis = flow_to_image(
            diff[0].permute(1, 2, 0).cpu().numpy(),
            convert_to_bgr=True,
        )

        cv2.imwrite(
            f"{path}flow_diff_layer{layer_idx}.jpg",
            diff_vis,
        )

        epe = torch.sum(
            (selected_flow - flow_gt_eval) ** 2,
            dim=1,
        ).sqrt()

        if val is not None:
            print(
                "EPE layer %d: %.3f"
                % (
                    layer_idx,
                    epe[0][val > 0].mean().cpu().item(),
                )
            )
        else:
            print(
                "EPE layer %d: %.3f"
                % (
                    layer_idx,
                    epe[0].mean().cpu().item(),
                )
            )

@torch.no_grad()
def demo_sintel(model, args, device=torch.device('cuda')):
    dstype = 'final'
    dataset = datasets.MpiSintel(split='training', dstype=dstype, root=args.paths['sintel'])
    image1, image2, flow_gt, _ = dataset[args.id]
    image1 = image1[None].to(device)
    image2 = image2[None].to(device)
    flow_gt = flow_gt[None].to(device)
    demo_data('sintel', args, model, image1, image2, flow_gt)

@torch.no_grad()
def demo_kitti(model, args, device=torch.device('cuda')):
    dstype = 'final'
    dataset = datasets.KITTI(split='training', root=args.paths['kitti'])
    image1, image2, flow_gt, val = dataset[args.id]

    image1 = image1[None].to(device)
    image2 = image2[None].to(device)
    flow_gt = flow_gt[None].to(device)
    val = val.to(device)

    demo_data('kitti', args, model, image1, image2, flow_gt, val)

@torch.no_grad()
def demo_spring(model, args, device=torch.device('cuda'), split='train'):
    dataset = datasets.SpringFlowDemoDataset(split=split, root=args.paths['spring'])
    if split == 'train' or split == 'val':
        image1, image2, flow_gt, _ = dataset[args.id]
    else:
        image1, image2,  _ = dataset[args.id]
        h, w = image1.shape[1:]
        flow_gt = torch.zeros((2, h, w))

    image1 = image1[None].to(device)
    image2 = image2[None].to(device)
    flow_gt = flow_gt[None].to(device)
    demo_data('spring', args, model, image1, image2, flow_gt)

@torch.no_grad()
def demo_layeredflow(model, args, device=torch.device('cuda')):
    dataset = datasets.LayeredFlow(root=args.paths['layeredflow'])

    print("LayeredFlow dataset length:", len(dataset))

    selected_index, scene_name, scene_sample_id = resolve_layeredflow_index(
        dataset,
        args,
    )

    image1, image2, coords, flow_gt, materials, layers = dataset[selected_index]

    image1 = image1[None].to(device)
    image2 = image2[None].to(device)

    save_name = f"layeredflow_scene{scene_name}_sample{scene_sample_id}"

    demo_data(
        save_name,
        args,
        model,
        image1,
        image2,
        flow_gt=None,
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--model', help='checkpoint path', type=str, required=True)
    parser.add_argument('--scale', help='input scale', type=int, default=0)
    parser.add_argument('--dataset', help='dataset type', type=str, required=True)
    parser.add_argument('--id', help='image id', type=int, default=0)
    parser.add_argument(
        '--scene',
        help='LayeredFlow scene/folder name, e.g. 6',
        type=str,
        default=None,
    )
    parser.add_argument(
        '--layer_idx',
        help='which predicted layer to visualize/evaluate',
        type=int,
        default=0,
    )

    parser.add_argument(
        '--save_all_layers',
        help='save visualizations for all predicted layers',
        action='store_true',
    )

    args = parse_args(parser)
    model = FlowSeek(args)

    load_ckpt_multihead(
        model,
        args.model,
    )

    model = model.cuda()
    model.eval()

    if args.dataset == 'sintel':
        demo_sintel(model, args)
    elif args.dataset == 'kitti':
        demo_kitti(model, args)  
    elif args.dataset == 'spring':
        demo_spring(model, args, split='train')
    elif args.dataset == 'layeredflow':
        demo_layeredflow(model, args)    

if __name__ == '__main__':
    main()