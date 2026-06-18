import argparse
import json
import os
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.append("core")

from flowseek import FlowSeek
from utils.flow_viz import flow_to_image


def load_json_as_namespace(cfg_path, model_path, scale=None, iters=None):
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    # config/parser.pyを使わず、JSONを直接Namespace化する
    args = SimpleNamespace(**cfg)

    # datapaths.jsonも読み、args.pathsとして追加する
    datapaths_path = "config/datapaths.json"
    if os.path.exists(datapaths_path):
        with open(datapaths_path, "r") as f:
            datapaths = json.load(f)
        if "paths" in datapaths:
            args.paths = datapaths["paths"]

    args.model = model_path

    # demo.pyと同じ発想でscale/itersを持たせる
    if not hasattr(args, "scale"):
        args.scale = 0
    if scale is not None:
        args.scale = scale

    if not hasattr(args, "iters"):
        args.iters = 4
    if iters is not None:
        args.iters = iters

    return args


def load_checkpoint_robust(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "model" in ckpt:
            state = ckpt["model"]
        else:
            state = ckpt
    else:
        state = ckpt

    try:
        model.load_state_dict(state, strict=True)
        return
    except RuntimeError:
        pass

    # DataParallel保存で "module." が付いている場合に対応
    stripped = {}
    for k, v in state.items():
        if k.startswith("module."):
            stripped[k[len("module."):]] = v
        else:
            stripped[k] = v

    model.load_state_dict(stripped, strict=True)


def read_image_as_tensor(path, device):
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise FileNotFoundError(f"画像を読み込めません: {path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float()
    tensor = tensor.unsqueeze(0).to(device)

    return tensor, img_rgb


def write_flo(path, flow):
    """
    Middlebury .flo形式で保存する。
    flow shape: [H, W, 2], float32
    """
    flow = flow.astype(np.float32)
    h, w, _ = flow.shape

    with open(path, "wb") as f:
        f.write(b"PIEH")
        np.array([w], dtype=np.int32).tofile(f)
        np.array([h], dtype=np.int32).tofile(f)
        flow.tofile(f)


@torch.no_grad()
def infer_flow(args, model, image1, image2):
    scale = args.scale

    img1 = F.interpolate(
        image1,
        scale_factor=2 ** scale,
        mode="bilinear",
        align_corners=False
    )

    img2 = F.interpolate(
        image2,
        scale_factor=2 ** scale,
        mode="bilinear",
        align_corners=False
    )

    output = model(img1, img2, iters=args.iters, test_mode=True)
    flow = output["flow"][-1]

    flow = F.interpolate(
        flow,
        scale_factor=0.5 ** scale,
        mode="bilinear",
        align_corners=False
    ) * (0.5 ** scale)

    return flow


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--cfg", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--image1", required=True)
    parser.add_argument("--image2", required=True)
    parser.add_argument("--outdir", default="demo_qualitatives/custom_pair")
    parser.add_argument("--scale", type=int, default=None)
    parser.add_argument("--iters", type=int, default=None)

    cli = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = load_json_as_namespace(
        cfg_path=cli.cfg,
        model_path=cli.model,
        scale=cli.scale,
        iters=cli.iters
    )

    model = FlowSeek(args)
    load_checkpoint_robust(model, cli.model)
    model.to(device)
    model.eval()

    image1, image1_rgb = read_image_as_tensor(cli.image1, device)
    image2, image2_rgb = read_image_as_tensor(cli.image2, device)

    if image1.shape != image2.shape:
        raise ValueError(
            f"image1とimage2のサイズが違います: "
            f"{tuple(image1.shape)} vs {tuple(image2.shape)}"
        )

    os.makedirs(cli.outdir, exist_ok=True)

    flow = infer_flow(args, model, image1, image2)

    flow_np = flow[0].permute(1, 2, 0).detach().cpu().numpy()

    flow_vis = flow_to_image(flow_np, convert_to_bgr=True)

    cv2.imwrite(
        os.path.join(cli.outdir, "image1.png"),
        cv2.cvtColor(image1_rgb, cv2.COLOR_RGB2BGR)
    )

    cv2.imwrite(
        os.path.join(cli.outdir, "image2.png"),
        cv2.cvtColor(image2_rgb, cv2.COLOR_RGB2BGR)
    )

    cv2.imwrite(
        os.path.join(cli.outdir, "flow_vis.png"),
        flow_vis
    )

    np.save(
        os.path.join(cli.outdir, "flow.npy"),
        flow_np
    )

    write_flo(
        os.path.join(cli.outdir, "flow.flo"),
        flow_np
    )

    print("Saved:")
    print(" ", os.path.join(cli.outdir, "image1.png"))
    print(" ", os.path.join(cli.outdir, "image2.png"))
    print(" ", os.path.join(cli.outdir, "flow_vis.png"))
    print(" ", os.path.join(cli.outdir, "flow.npy"))
    print(" ", os.path.join(cli.outdir, "flow.flo"))
    print("flow shape:", flow_np.shape)
    print("u min/max:", flow_np[..., 0].min(), flow_np[..., 0].max())
    print("v min/max:", flow_np[..., 1].min(), flow_np[..., 1].max())


if __name__ == "__main__":
    main()
