"""Side-by-side GIF of a dataset clip and the user's recording of the same sign, plus a text report
of which hand is active and the coordinate ranges of both.

    python scripts/compare.py --sign <word> [--recording recordings/<word>_<time>.npy]

Without --recording every recordings/<word>_*.npy is compared. GIFs go to reports/compare/.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from asl import config, skeleton  # noqa: E402
from asl.capture import load_recording  # noqa: E402
from asl.landmarks_io import LEFT_HAND, POSE, RIGHT_HAND, load_clip  # noqa: E402

L_SHOULDER, R_SHOULDER = 11, 12


def hand_frames(clip):
    return {"left_hand": int((~np.isnan(clip[:, LEFT_HAND, 0]).all(1)).sum()),
            "right_hand": int((~np.isnan(clip[:, RIGHT_HAND, 0]).all(1)).sum())}


def describe(clip):
    """Active hand, its image side relative to the body, and coordinate ranges."""
    counts = hand_frames(clip)
    active = max(counts, key=counts.get) if any(counts.values()) else None
    sh = clip[:, POSE][:, [L_SHOULDER, R_SHOULDER], :2]
    mid_x = np.nanmedian((sh[:, 0, 0] + sh[:, 1, 0]) / 2)
    width = np.nanmedian(np.abs(sh[:, 0, 0] - sh[:, 1, 0]))
    out = {"frames": len(clip), **counts, "active": active,
           "shoulder_mid_x": float(mid_x), "shoulder_width": float(width),
           # pose 11 is the person's left shoulder; it is on the image right when the image is not mirrored
           "pose_left_shoulder_on_image": "right" if np.nanmedian(sh[:, 0, 0]) > mid_x else "left"}
    if active:
        block = LEFT_HAND if active == "left_hand" else RIGHT_HAND
        pts = clip[:, block, :2].reshape(-1, 2)
        pts = pts[~np.isnan(pts).any(1)]
        out["active_hand_side"] = "right" if np.median(pts[:, 0]) > mid_x else "left"
        out["active_hand_x"] = (float(pts[:, 0].min()), float(pts[:, 0].max()))
        out["active_hand_y"] = (float(pts[:, 1].min()), float(pts[:, 1].max()))
        # Report only, not model input. Relative to the body in shoulder widths: x is comparable across
        # cameras, y is not (no aspect correction here; features.extract does that).
        rel = (np.median(pts, axis=0) - [mid_x, np.nanmedian(sh[:, :, 1])]) / max(width, 1e-6)
        out["active_hand_rel_body"] = (round(float(rel[0]), 2), round(float(rel[1]), 2))
    return out


def best_dataset_clip(sign, active=None):
    """Local dataset clip of the sign with the most hand frames; prefers the same active hand as
    `active` ("left_hand" / "right_hand"), so handedness differences between people don't look like
    a mirroring problem."""
    train = pd.read_csv(config.SAMPLE_DIR / "train.csv")
    paths = [config.SAMPLE_DIR / p for p in train.loc[train["sign"] == sign, "path"]]
    paths = [p for p in paths if p.exists()]
    if not paths:
        raise SystemExit(f"No dataset clip for '{sign}' in data/sample")
    same = [p for p in paths if active and describe(load_clip(p))["active"] == active]
    return max(same or paths, key=lambda p: max(hand_frames(load_clip(p)).values()))


def save_side_by_side(a, b, titles, out_path, fps=15):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    n = max(len(a), len(b))
    ia = np.linspace(0, len(a) - 1, n).round().astype(int)
    ib = np.linspace(0, len(b) - 1, n).round().astype(int)
    bounds = [skeleton.display_bounds(a), skeleton.display_bounds(b)]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.4), dpi=90)
    fig.patch.set_facecolor(skeleton.BACKGROUND)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.9, wspace=0.05)

    def draw(t):
        for ax, clip, idx, (x0, x1, y0, y1), title in zip(axes, (a, b), (ia, ib), bounds, titles):
            ax.clear()
            ax.set_xlim(x0, x1)
            ax.set_ylim(y1, y0)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(title, fontsize=11, color=skeleton.INK, loc="left")
            skeleton.draw_frame(ax, clip[idx[t]])

    FuncAnimation(fig, draw, frames=n).save(str(out_path), writer=PillowWriter(fps=fps))
    plt.close(fig)


def print_report(name, d):
    print(f"  {name}: {d['frames']} frames, left_hand={d['left_hand']} right_hand={d['right_hand']}, "
          f"active={d['active']}")
    if d["active"]:
        print(f"    active hand on image {d['active_hand_side']} of body; x {d['active_hand_x'][0]:.2f}..{d['active_hand_x'][1]:.2f}, "
              f"y {d['active_hand_y'][0]:.2f}..{d['active_hand_y'][1]:.2f}; relative to shoulders (in shoulder widths) {d['active_hand_rel_body']}")
    print(f"    shoulders: mid x {d['shoulder_mid_x']:.2f}, width {d['shoulder_width']:.2f}; "
          f"pose left shoulder on image {d['pose_left_shoulder_on_image']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sign", required=True)
    ap.add_argument("--recording", type=Path, action="append")
    ap.add_argument("--dataset", type=Path, help="dataset parquet (default: best local clip with the same active hand)")
    args = ap.parse_args()

    recs = args.recording or sorted(config.RECORDINGS_DIR.glob(f"{args.sign}_*.npy"))
    if not recs:
        raise SystemExit(f"No recordings/{args.sign}_*.npy yet")
    out_dir = config.REPORTS_DIR / "compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"sign '{args.sign}'")
    for rec in recs:
        clip, meta = load_recording(rec)
        info = describe(clip)
        ds_path = args.dataset or best_dataset_clip(args.sign, info["active"])
        ds = load_clip(ds_path)
        ds_info = describe(ds)
        print_report(f"dataset {ds_path.parent.name}/{ds_path.name}", ds_info)
        print_report(f"recording {rec.name} ({meta.get('fps', '?')} fps, mirror_input={meta.get('mirror_input')})", info)
        same = info["active"] == ds_info["active"]
        side_same = info.get("active_hand_side") == ds_info.get("active_hand_side")
        print(f"    active hand vs dataset clip: {'MATCH' if same else 'DIFFERENT'}; "
              f"side of body on the image: {'MATCH' if side_same else 'DIFFERENT'}")
        out = out_dir / f"{rec.stem}.gif"
        save_side_by_side(ds, clip, [f"dataset: {args.sign}", f"you: {rec.stem}"], out)
        print(f"    gif: {out}")


if __name__ == "__main__":
    main()
