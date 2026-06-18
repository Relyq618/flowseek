from pathlib import Path
import argparse
import os
import re
import subprocess
import sys


IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def natural_key(path: Path):
    """
    ファイル名を自然順に並べるためのキー。
    例:
        2_rectified.png  < 10_rectified.png
        frame_0002.png   < frame_0010.png
    """
    text = path.name
    parts = re.split(r"(\d+)", text)
    key = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return key


def collect_image_dirs(root: Path):
    """
    root以下を再帰的に探索し、
    画像ファイルを2枚以上含むディレクトリだけを返す。
    """
    dirs = {}

    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            dirs.setdefault(p.parent, []).append(p)

    result = []
    for d, images in dirs.items():
        images = sorted(images, key=natural_key)
        if len(images) >= 2:
            result.append((d, images))

    result.sort(key=lambda x: str(x[0]))
    return result


def safe_rel_dir(path: Path, root: Path):
    """
    出力ディレクトリ名に使いやすい相対パス文字列を作る。
    """
    rel = path.relative_to(root)
    return str(rel).replace("/", "__")


def run_pair(
    infer_script: str,
    cfg: str,
    model: str,
    image1: Path,
    image2: Path,
    outdir: Path,
    scale: int,
    iters,
):
    cmd = [
        sys.executable,
        infer_script,
        "--cfg", cfg,
        "--model", model,
        "--image1", str(image1),
        "--image2", str(image2),
        "--outdir", str(outdir),
        "--scale", str(scale),
    ]

    if iters is not None:
        cmd += ["--iters", str(iters)]

    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        default="/home/amemiyq/layeredflow/val",
        help="LayeredFlow val directory"
    )
    parser.add_argument(
        "--out-root",
        default="demo_qualitatives/layeredflow_val_all",
        help="Output root directory"
    )
    parser.add_argument(
        "--cfg",
        default="config/eval/flowseek-T.json"
    )
    parser.add_argument(
        "--model",
        default="weights/flowseek_T_TartanCT_TSKH.pth"
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=-1,
        help="Inference scale. -1 means half resolution, -2 means quarter resolution."
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=None
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="For test run. Stop after N pairs."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list target pairs, do not run inference."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs."
    )

    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    out_root = Path(args.out_root)

    if not root.exists():
        raise FileNotFoundError(f"root not found: {root}")

    if not Path("infer_pair.py").exists():
        raise FileNotFoundError(
            "infer_pair.py が ~/flowseek に見つかりません。"
            "先に任意画像ペア推論用 infer_pair.py を作成してください。"
        )

    image_dirs = collect_image_dirs(root)

    print("root:", root)
    print("out_root:", out_root)
    print("num image dirs:", len(image_dirs))
    print("scale:", args.scale)
    print()

    total_pairs = 0
    plan = []

    for d, images in image_dirs:
        rel_name = safe_rel_dir(d, root)

        for i in range(len(images) - 1):
            image1 = images[i]
            image2 = images[i + 1]

            pair_name = f"{i:04d}_{image1.stem}_to_{image2.stem}"
            outdir = out_root / rel_name / pair_name

            plan.append((d, image1, image2, outdir))
            total_pairs += 1

    print("total pairs:", total_pairs)
    print()

    if args.dry_run:
        for idx, (d, image1, image2, outdir) in enumerate(plan):
            print("=" * 80)
            print(f"[dry-run {idx + 1}/{len(plan)}]")
            print("dir   :", d)
            print("image1:", image1)
            print("image2:", image2)
            print("outdir:", outdir)

            if args.limit is not None and idx + 1 >= args.limit:
                break

        print()
        print("Dry run finished.")
        return

    out_root.mkdir(parents=True, exist_ok=True)

    done = 0
    skipped = 0

    for idx, (d, image1, image2, outdir) in enumerate(plan):
        if args.limit is not None and done >= args.limit:
            break

        flow_vis = outdir / "flow_vis.png"
        flow_npy = outdir / "flow.npy"

        if not args.overwrite and flow_vis.exists() and flow_npy.exists():
            skipped += 1
            print("=" * 80)
            print(f"[skip {idx + 1}/{len(plan)}] already exists")
            print("outdir:", outdir)
            continue

        print("=" * 80)
        print(f"[run {idx + 1}/{len(plan)}]")
        print("dir   :", d)
        print("image1:", image1)
        print("image2:", image2)
        print("outdir:", outdir)

        outdir.mkdir(parents=True, exist_ok=True)

        try:
            run_pair(
                infer_script="infer_pair.py",
                cfg=args.cfg,
                model=args.model,
                image1=image1,
                image2=image2,
                outdir=outdir,
                scale=args.scale,
                iters=args.iters,
            )

            with open(outdir / "info.txt", "w") as f:
                f.write(f"root: {root}\n")
                f.write(f"source_dir: {d}\n")
                f.write(f"image1: {image1}\n")
                f.write(f"image2: {image2}\n")
                f.write(f"cfg: {args.cfg}\n")
                f.write(f"model: {args.model}\n")
                f.write(f"scale: {args.scale}\n")
                if args.iters is not None:
                    f.write(f"iters: {args.iters}\n")

            done += 1

        except subprocess.CalledProcessError as e:
            print("ERROR: inference failed")
            print("image1:", image1)
            print("image2:", image2)
            print("return code:", e.returncode)
            print("If this is OOM, rerun with --scale -2.")
            raise

    print("=" * 80)
    print("Finished.")
    print("done   :", done)
    print("skipped:", skipped)
    print("out_root:", out_root)


if __name__ == "__main__":
    main()
