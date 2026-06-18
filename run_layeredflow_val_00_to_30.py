from pathlib import Path
import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        default="/home/amemiyq/layeredflow/val",
        help="LayeredFlow val directory"
    )
    parser.add_argument(
        "--out-root",
        default="demo_qualitatives/layeredflow_val_00_to_30",
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
        help="For test run. Stop after N scenes."
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
        )

    scene_dirs = sorted(
        [p for p in root.iterdir() if p.is_dir()],
        key=lambda p: int(p.name) if p.name.isdigit() else p.name
    )

    targets = []

    for scene_dir in scene_dirs:
        image1 = scene_dir / "0_0.png"
        image2 = scene_dir / "3_0.png"

        if image1.exists() and image2.exists():
            outdir = out_root / scene_dir.name / "0_0_to_3_0"
            targets.append((scene_dir, image1, image2, outdir))
        else:
            print("[missing]", scene_dir)
            if not image1.exists():
                print("  missing:", image1)
            if not image2.exists():
                print("  missing:", image2)

    print("root:", root)
    print("out_root:", out_root)
    print("num scene dirs:", len(scene_dirs))
    print("num valid pairs:", len(targets))
    print("scale:", args.scale)
    print()

    if args.dry_run:
        for idx, (scene_dir, image1, image2, outdir) in enumerate(targets):
            print("=" * 80)
            print(f"[dry-run {idx + 1}/{len(targets)}]")
            print("scene :", scene_dir.name)
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

    for idx, (scene_dir, image1, image2, outdir) in enumerate(targets):
        if args.limit is not None and done >= args.limit:
            break

        flow_vis = outdir / "flow_vis.png"
        flow_npy = outdir / "flow.npy"

        if not args.overwrite and flow_vis.exists() and flow_npy.exists():
            skipped += 1
            print("=" * 80)
            print(f"[skip {idx + 1}/{len(targets)}] already exists")
            print("scene :", scene_dir.name)
            print("outdir:", outdir)
            continue

        print("=" * 80)
        print(f"[run {idx + 1}/{len(targets)}]")
        print("scene :", scene_dir.name)
        print("image1:", image1)
        print("image2:", image2)
        print("outdir:", outdir)

        outdir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            "infer_pair.py",
            "--cfg", args.cfg,
            "--model", args.model,
            "--image1", str(image1),
            "--image2", str(image2),
            "--outdir", str(outdir),
            "--scale", str(args.scale),
        ]

        if args.iters is not None:
            cmd += ["--iters", str(args.iters)]

        try:
            subprocess.run(cmd, check=True)

            with open(outdir / "info.txt", "w") as f:
                f.write(f"scene: {scene_dir.name}\n")
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
            print("scene :", scene_dir.name)
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
