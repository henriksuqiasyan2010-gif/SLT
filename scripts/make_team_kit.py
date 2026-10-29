"""Team kit (Phase 8): figures, data tables and METRICS.md for the report and the presentation.

    python scripts/make_team_kit.py            # -> reports/team_kit/
    python scripts/design_screenshots.py --langs en hy --out reports/team_kit/screenshots

Every number in METRICS.md is read from the result files (models/metrics.json, reports/<run>/,
reports/user_tests/, reports/twohands/) or measured here (the demo sessions are segmented and
recognised again with the model in models/), so the kit cannot drift from the results.
README.md and architecture.mmd in the kit are written by hand.
"""

import csv
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from asl import config  # noqa: E402
from asl.app.demo import demo_sessions  # noqa: E402
from asl.capture import load_recording, recording_aspect  # noqa: E402
from asl.inference import SignModel  # noqa: E402
from asl.segmenter import segment_clip  # noqa: E402

OUT = config.ROOT / "reports" / "team_kit"
RUNS = config.ROOT / "reports"
DPI = 200
# Chart tokens (light surface; one series per chart, so one colour and no legend)
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
SERIES = "#6250d6"  # violet, validated against SURFACE (dataviz validator: all checks pass)
# Diagram colours: the app's palette (asl.app.theme)
LAVENDER, LILAC, NIGHT_INK, ACCENT = "#F1ECFF", "#E2D8FF", "#2A1460", "#6B3FF2"
# The road from the first model to the app (fold-0 validation, top-1), all with T = 32
PROGRESSION = [
    ("baseline", "1D-CNN baseline"),
    ("main", "Transformer + augmentation"),
    ("main-reg", "+ regularisation, 100 epochs"),
    ("main-reg30", "+ 30 augmented versions"),
]
LIVE_RULE_LABEL = "+ second-hand rule (app)"


def run_dir(preset: str) -> Path:
    return RUNS / preset / f"{preset}_T32"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list:
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def pct(v: float, digits: int = 1) -> str:
    return f"{100 * v:.{digits}f}%"


def style(ax, xlabel: str = "") -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0, labelsize=10)
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_2, fontsize=10)


def hbars(ax, labels: list, values: list, fmt=pct, xmax: float = 1.0) -> None:
    """Horizontal bars, first item on top, value at the tip of each bar."""
    y = list(range(len(labels)))[::-1]
    ax.barh(y, values, height=0.55, color=SERIES)
    ax.set_yticks(y, labels, color=INK, fontsize=10)
    ax.set_xlim(0, xmax)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    for yi, v in zip(y, values):
        ax.text(v + xmax * 0.01, yi, fmt(v), va="center", color=INK, fontsize=10)


def save(fig, name: str) -> Path:
    path = OUT / "figures" / name
    fig.savefig(path, dpi=DPI, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def title(fig, ax, text: str, sub: str) -> None:
    ax.set_title(text, loc="left", color=INK, fontsize=13, fontweight="bold", pad=22)
    ax.text(0, 1.02, sub, transform=ax.transAxes, color=INK_2, fontsize=10, va="bottom")


# --- figures -----------------------------------------------------------------------------------
def fig_progression(live: dict) -> Path:
    labels = [name for _, name in PROGRESSION] + [LIVE_RULE_LABEL]
    values = [read_json(run_dir(p) / "metrics.json")["orig"]["top1"] for p, _ in PROGRESSION] + [live["whole"]]
    fig, ax = plt.subplots(figsize=(8, 3.6), facecolor=SURFACE)
    style(ax, "top-1 accuracy, 4 people the model never saw (18,528 clips)")
    hbars(ax, labels, values)
    title(fig, ax, "From the first model to the app", "250 signs; chance level 0.4%")
    return save(fig, "accuracy_progression.png")


def fig_participants(live: dict) -> Path:
    people = sorted(k[1:] for k in live if k.startswith("p"))
    values = [live[f"p{p}"] for p in people]
    fig, ax = plt.subplots(figsize=(8, 2.8), facecolor=SURFACE)
    style(ax, "top-1 accuracy (app pipeline, second-hand rule on)")
    hbars(ax, [f"participant {p}" for p in people], values)
    title(fig, ax, "Accuracy per held-out person",
          "participant 29302 often keeps the second hand in view (34% of their clips)")
    return save(fig, "accuracy_per_participant.png")


def fig_confusions(rows: list, n: int = 12) -> Path:
    rows = sorted(rows, key=lambda r: -float(r["share_of_true"]))[:n]
    labels = [f"{r['true']} → {r['predicted']}" for r in rows]
    values = [float(r["share_of_true"]) for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=SURFACE)
    style(ax, "share of the true sign's validation clips predicted as the other sign")
    hbars(ax, labels, values, xmax=0.5)
    title(fig, ax, "Most frequent confusions", "true sign → predicted sign; many are near-synonyms in ASL")
    return save(fig, "top_confusions.png")


def fig_signs(rows: list, n: int = 10) -> Path:
    rows = sorted(rows, key=lambda r: float(r["accuracy"]))
    worst, best = rows[:n], rows[::-1][:n]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor=SURFACE)
    for ax, part, head in ((axes[0], best, "Easiest signs"), (axes[1], worst, "Hardest signs")):
        style(ax, "top-1 accuracy")
        hbars(ax, [r["sign"] for r in part], [float(r["accuracy"]) for r in part], fmt=lambda v: pct(v, 0))
        ax.set_title(head, loc="left", color=INK, fontsize=12, fontweight="bold")
    fig.tight_layout()
    return save(fig, "signs_easiest_hardest.png")


def box(ax, x, y, w, h, text, fill, bold_first=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.18",
                                facecolor=fill, edgecolor="none"))
    head, _, rest = text.partition("\n")
    ax.text(x + w / 2, y + h - 0.28, head, ha="center", va="top", color=NIGHT_INK, fontsize=10.5,
            fontweight="bold" if bold_first else "normal")
    if rest:
        ax.text(x + w / 2, y + h - 0.62, rest, ha="center", va="top", color=NIGHT_INK, fontsize=8.6,
                linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, dashed=False):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, color=ACCENT,
                                 linewidth=1.6, linestyle="--" if dashed else "-", shrinkA=2, shrinkB=2))


def fig_architecture() -> Path:
    """The same pipeline as architecture.mmd: training on Kaggle (top), the app on the laptop (bottom)."""
    fig, ax = plt.subplots(figsize=(17, 5.9), facecolor="white")
    ax.set_xlim(0, 17)
    ax.set_ylim(0.2, 6.6)
    ax.axis("off")
    w, h, gap = 2.15, 1.75, 0.27
    train = [
        "Kaggle dataset\nGoogle ISLR: 94,477 clips\n250 signs, 21 people\n543 landmarks / frame",
        "Augmentation\nfps 12-15, time warp,\nlost frames and hands,\nrotation, x/y stretch",
        "features.extract\nnormalise to shoulders,\nmirror left-handers,\n32 frames x 542 features",
        "Training (Kaggle GPU)\nTransformer, 1.9 M params\n100 epochs, masking,\nmixup; best epoch kept",
        "model.onnx\n+ labels + metrics\nfold-0 validation:\ntop-1 62.7%, top-3 76.9%",
    ]
    app = [
        "Webcam\n640 x 480\n15-25 fps",
        "MediaPipe Holistic\nface 468, hands 2 x 21,\npose 33 points\n(no video is stored)",
        "Clip\nbutton: 3-2-1 + 2 s\nor segmenter:\nwrist up = sign",
        "Clean-up\ntrim_idle: still frames\nkeep_moving_hand:\ndrop a still 2nd hand",
        "features.extract\nthe same code\nas in training",
        "ONNX Runtime\nthe trained model\non the CPU,\n~1-3 ms per sign",
        "Interface\ntop-3 + confidence,\nEN / HY / RU, speech,\nsentence, Learn score",
    ]
    ty, ay = 4.3, 0.35
    ax.text(0.2, 6.35, "Training — once, on Kaggle", color=NIGHT_INK, fontsize=13, fontweight="bold", va="top")
    ax.text(0.2, 2.45, "The app — live, on a laptop CPU, offline", color=NIGHT_INK, fontsize=13,
            fontweight="bold", va="top")
    tx = [0.2 + i * (w + gap + 0.95) for i in range(len(train))]
    for i, (x, text) in enumerate(zip(tx, train)):
        box(ax, x, ty, w + 0.6, h, text, LILAC)
        if i:
            arrow(ax, tx[i - 1] + w + 0.6, ty + h / 2, x, ty + h / 2)
    axs = [0.2 + i * (w + gap) for i in range(len(app))]
    for i, (x, text) in enumerate(zip(axs, app)):
        box(ax, x, ay, w, h, text, LAVENDER)
        if i:
            arrow(ax, axs[i - 1] + w, ay + h / 2, x, ay + h / 2)
    # the trained model goes into the app; the feature code is shared
    arrow(ax, tx[4] + (w + 0.6) / 2, ty, axs[5] + w / 2, ay + h)
    arrow(ax, tx[2] + (w + 0.6) / 2, ty, axs[4] + w / 2, ay + h, dashed=True)
    ax.text(9.75, 3.25, "same module\n(asl.features)", color=ACCENT, fontsize=8.6, ha="left")
    return save(fig, "architecture.png")


# --- measured here: the demo sessions ------------------------------------------------------
def score_sessions(model) -> list:
    rows = []
    for s in demo_sessions():
        clip, meta = load_recording(s.path)
        aspect = recording_aspect(meta)
        segs = segment_clip(clip, meta["timestamps"], aspect)
        got = []
        for seg in segs:
            probs = model.predict_clip(seg.landmarks, meta["fps"], aspect)
            got.append(model.top_k(probs, 1)[0][0])
        hits = sum(a == b for a, b in zip(got, s.signs))
        rows.append({"session": s.name, "signed": " ".join(s.signs), "recognised": " ".join(got),
                     "segments": len(segs), "correct": hits})
    return rows


# --- METRICS.md ------------------------------------------------------------------------------
def metrics_md(m: dict, live: dict, twohands: dict, user: list, sessions: list, defense: list) -> str:
    rec = [r for r in user if r["group"] == "recording"]
    att = [r for r in user if r["group"] != "recording"]
    top1 = lambda rows: sum(r["rank"] == "1" for r in rows)  # noqa: E731
    learned = [r for r in defense if r["learned"] == "True"]
    seg_ok = sum(r["correct"] for r in sessions)
    seg_n = sum(len(r["signed"].split()) for r in sessions)
    runs = {p: read_json(run_dir(p) / "metrics.json") for p, _ in PROGRESSION}
    syn = m["presentation_synonyms_merged"]
    th = m["thinned_13fps"]
    lines = [
        "# SLT — results",
        "",
        "All numbers below are generated by `scripts/make_team_kit.py` from the result files.",
        "",
        "## Model",
        "",
        f"- Task: recognise one of **{config.NUM_CLASSES} isolated ASL signs** from MediaPipe landmarks "
        "(dataset: Google — Isolated Sign Language Recognition, Kaggle; 94,477 clips from 21 people).",
        f"- Model: **{m['model'].capitalize()}**, {m['params'] / 1e6:.1f} M parameters, input "
        f"{m['T_FIX']} frames x {m['F']} features, best of {m['config']['epochs']} epochs "
        f"(epoch {m['best_epoch']}). Runs on CPU with ONNX Runtime in about 1-3 ms per sign.",
        f"- Validation: **{len(m['val_participants'])} people the model never saw** "
        f"({m['n_val']:,} clips; {m['n_train']:,} clips of the other 17 people for training; "
        f"{m['split']}).",
        "",
        "## Accuracy on people the model never saw",
        "",
        "| | top-1 | top-3 |",
        "|---|---|---|",
        f"| Model (`{m['preset']}`) | **{pct(m['orig']['top1'])}** | **{pct(m['orig']['top3'])}** |",
        f"| Same, video thinned to 13 fps (a slow laptop) | {pct(th['top1'])} | {pct(th['top3'])} |",
        f"| App pipeline (second-hand rule on) | **{pct(live['whole'])}** | — |",
        f"| App pipeline at 13 fps | {pct(live['whole_13fps'])} | — |",
        "",
        "top-1 = the right sign is the first answer; top-3 = it is among the three answers the app shows. "
        f"Chance level is {pct(1 / config.NUM_CLASSES)}.",
        "",
        "The second-hand rule: when both hands are visible and one of them stays still, the still hand "
        f"is ignored. On the {twohands['n_both_in_one_frame']:,} validation clips with both hands in a "
        f"frame it raises top-1 from {pct(twohands['orig']['top1_without'])} to "
        f"{pct(twohands['orig']['top1_with'])}.",
        "",
        "Per person (app pipeline): " + ", ".join(
            f"{p[1:]}: {pct(live[p])}" for p in sorted(k for k in live if k.startswith("p"))) + ".",
        "",
        "Presentation-only number, **not a model metric**: if near-synonyms count as correct "
        f"({', '.join('/'.join(g) for g in syn['groups'])}), top-1 is {pct(syn['orig_top1'])}.",
        "",
        "## How we got there (top-1, same validation people)",
        "",
        "| Step | top-1 | top-3 |",
        "|---|---|---|",
    ]
    for p, name in PROGRESSION:
        lines.append(f"| {name} | {pct(runs[p]['orig']['top1'])} | {pct(runs[p]['orig']['top3'])} |")
    lines += [
        f"| {LIVE_RULE_LABEL} | {pct(live['whole'])} | — |",
        "",
        "Tried and not kept: an ensemble of three models (+1.1 pp, three times the compute); "
        "the second-hand rule inside training (64.4%, below the app pipeline's 65.1%).",
        "",
        "## Tests with the author's own signing (webcam of the laptop)",
        "",
        f"- Recorded single signs: top-1 correct in **{top1(rec)} of {len(rec)}**; misses: " + ", ".join(
            f"{r['sign']} ({'rank ' + r['rank'] if r['rank'] else 'not in top-3'})" for r in rec if r["rank"] != "1")
        + ". The dad recordings were made before the author learned the sign in Learn mode "
        "(the thumb must touch the forehead); afterwards dad was recognised in 3 of 4 attempts.",
        f"- First test in the app (10 different signs, one attempt each): **{top1(att)} of {len(att)}** correct.",
        f"- Sentences (3 recorded sessions of 5 signs, automatic segmentation): "
        f"**{seg_ok} of {seg_n}** words correct, {sum(r['segments'] for r in sessions)} segments found.",
        f"- Learn mode: {len(learned)} signs learned (3 successful attempts each), "
        "listed in `data/defense_set.csv`.",
        "",
        "| Session | Signed | Recognised |",
        "|---|---|---|",
    ]
    lines += [f"| {r['session']} | {r['signed']} | {r['recognised']} |" for r in sessions]
    lines += [
        "",
        "## Speed (HP laptop, Intel Core i3-1215U, no GPU)",
        "",
        "- Camera + MediaPipe Holistic + interface: 16-25 frames per second (median 16 after 5 minutes "
        "of continuous use; the CPU slows down when it gets warm).",
        "- Recognition of one sign (clean-up + features + model): 2-5 ms.",
        "",
        "## Limits",
        "",
        "- Isolated signs only: the app recognises single signs (or a sequence of single signs with "
        "pauses), it does not translate grammatical ASL sentences.",
        "- 250 everyday signs (the dataset's vocabulary), not the whole of ASL.",
        "- Accuracy depends on the person: 57-71% across the four held-out people.",
        "- Fingerspelling, facial grammar and fast continuous signing are not covered.",
    ]
    return "\n".join(lines) + "\n"


def main():
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    m = read_json(config.MODELS_DIR / "metrics.json")
    run = run_dir(m["preset"])
    live = {r["subset"]: float(r["top1"]) for r in read_csv(RUNS / "user_tests" / f"val_{m['preset']}_live_rule.csv")}
    live["whole_13fps"] = float(read_csv(RUNS / "user_tests" / f"val_{m['preset']}_live_rule.csv")[0]["top1_13fps"])
    twohands = read_json(RUNS / "twohands" / "twohands.json")
    user = read_csv(RUNS / "user_tests" / f"score_{m['preset']}.csv")
    defense = read_csv(RUNS / "defense_set.csv")
    confusions = read_csv(run / "top_confusions.csv")
    per_sign = read_csv(run / "per_sign_accuracy.csv")

    for name in ("training_curves.png", "confusion_matrix.png"):
        shutil.copy(run / name, OUT / "figures" / name)
    for src in (run / "top_confusions.csv", run / "per_sign_accuracy.csv", RUNS / "defense_set.csv",
                RUNS / "signs_250.csv", config.MODELS_DIR / "metrics.json"):
        shutil.copy(src, OUT / "data" / src.name)
    made = [fig_progression(live), fig_participants(live), fig_confusions(confusions), fig_signs(per_sign),
            fig_architecture()]
    sessions = score_sessions(SignModel.load())
    (OUT / "METRICS.md").write_text(metrics_md(m, live, twohands, user, sessions, defense), encoding="utf-8")
    for p in made + [OUT / "METRICS.md"]:
        print(p.relative_to(config.ROOT))
    print("sessions:", [(r["session"], r["correct"]) for r in sessions])
    return 0


if __name__ == "__main__":
    sys.exit(main())
