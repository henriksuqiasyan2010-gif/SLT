"""Kaggle kernel asl-train (GPU): train a sign classifier on the asl-preprocess output.

PRESET (set by scripts/push_kernel.py when the kernel is pushed):
  baseline    small 1D-CNN on the precomputed features, T = 64 and T = 32 one after another
  main-trial  1D-CNN + Transformer, on-the-fly augmentation + extract(), 2 epochs: epoch time,
              GPU utilisation and the share of time the GPU waits for data
  main        the same model, full training with early stopping

Every model is evaluated twice on the held-out participants: on the original clips and on the
clips thinned to EVAL_FPS frames per second (the user's camera runs at 12-15 fps).
Outputs per run in /kaggle/working/<preset>_T<T>/: model.onnx, labels.json, metrics.json,
confusion_matrix.png, per_sign_accuracy.csv, top_confusions.csv, training_curves.png.
"""

PRESET = "baseline"  # replaced by scripts/push_kernel.py

PRESETS = {
    "baseline": dict(model="cnn", t_values=[64, 32], epochs=20, batch=256, lr=2e-3, wd=1e-4,
                     label_smoothing=0.0, augment=False, patience=0, warmup_epochs=1),
    "main-trial": dict(model="transformer", t_values=[64], epochs=2, batch=256, lr=1e-3, wd=0.05,
                       label_smoothing=0.1, augment=True, patience=0, warmup_epochs=1),
    "main": dict(model="transformer", t_values=[64], epochs=60, batch=256, lr=1e-3, wd=0.05,
                 label_smoothing=0.1, augment=True, patience=10, warmup_epochs=2),
}
SEED = 42
N_FOLDS = 5
VAL_FOLD = 0
EVAL_FPS = 13.0
NUM_WORKERS = 4
ONNX_PARITY_SAMPLES = 100
ONNX_PARITY_ATOL = 1e-4
ONNX_OPSET = 17
GPU_SAMPLE_S = 1.0
# Augmentation (main): probabilities and ranges. Applied to raw landmarks before extract().
AUG = dict(p_fps=0.5, fps_range=(12.0, 15.0), p_warp=0.5, warp_speed=(0.6, 1.6),
           p_drop=0.5, drop_max=0.15, p_hand_gap=0.3, hand_gap=(0.1, 0.3),
           p_rotate=0.5, rotate_deg=15.0, p_stretch=0.8, stretch=0.10)

import json  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def find_one(pattern, root="/kaggle/input"):
    hits = sorted(Path(root).glob(pattern))
    if not hits:
        raise FileNotFoundError(f"{root}/{pattern}")
    return hits[0]


# Kaggle unpacks the asl-src zip without its top folder, so rebuild the `asl` package under /tmp.
SRC_DIR = Path("/tmp/asl_src")
if Path("/kaggle/input").exists():
    shutil.copytree(find_one("**/landmarks_io.py").parent, SRC_DIR / "asl", dirs_exist_ok=True)
sys.path.insert(0, str(SRC_DIR))
from asl import config, features as F  # noqa: E402

OUT_ROOT = Path(os.environ.get("ASL_OUT", "/kaggle/working"))
PRE_DIR = Path(os.environ["ASL_PRE"]) if "ASL_PRE" in os.environ else find_one("**/X64.npy").parent
COMP_DIR = Path(os.environ["ASL_COMP"]) if "ASL_COMP" in os.environ else \
    find_one("**/sign_to_prediction_index_map.json").parent


def ensure_onnx():
    try:
        import onnx  # noqa: F401
        import onnxruntime  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnx", "onnxruntime"], check=True)


ensure_onnx()
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as TF  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Data ------------------------------------------------------------------------------------

class Data:
    def __init__(self):
        self.meta = json.loads((PRE_DIR / "meta.json").read_text())
        if self.meta["FEATURES_VERSION"] != F.FEATURES_VERSION:
            raise RuntimeError(f"preprocess FEATURES_VERSION {self.meta['FEATURES_VERSION']} != code {F.FEATURES_VERSION}")
        self.y = np.load(PRE_DIR / "y.npy").astype(np.int64)
        self.groups = np.load(PRE_DIR / "groups.npy")
        self.offsets = np.load(PRE_DIR / "raw_offsets.npy")
        labels = json.loads((COMP_DIR / "sign_to_prediction_index_map.json").read_text())
        self.signs = {v: k for k, v in labels.items()}
        self.n_classes = len(labels)
        from sklearn.model_selection import GroupKFold
        folds = list(GroupKFold(n_splits=N_FOLDS).split(np.zeros(len(self.y)), self.y, self.groups))
        self.train_idx, self.val_idx = np.sort(folds[VAL_FOLD][0]), np.sort(folds[VAL_FOLD][1])
        self.val_participants = sorted(int(g) for g in np.unique(self.groups[self.val_idx]))

    def features(self, t):
        return np.load(PRE_DIR / f"X{t}.npy", mmap_mode="r"), np.load(PRE_DIR / f"mask{t}.npy")


_RAW = None


def raw_clip(i):
    """(T, 130, 2) raw subset x, y of clip i; the arrays are opened lazily once per process."""
    global _RAW
    if _RAW is None:
        _RAW = (np.load(PRE_DIR / "raw_xy.npy", mmap_mode="r"), np.load(PRE_DIR / "raw_offsets.npy"))
    xy, off = _RAW
    return np.asarray(xy[off[i]:off[i + 1]], dtype=np.float32)


def to_clip(xy):
    """(T, 130, 2) subset x, y -> (T, 543, 3) clip for extract() (everything else NaN)."""
    clip = np.full((len(xy), 543, 3), np.nan, np.float32)
    clip[:, F.SUBSET, :2] = xy
    return clip


def thin_index(n, fps_dst, fps_src=config.DATASET_FPS_ASSUMED):
    """Frames kept when an fps_src clip is recorded at fps_dst instead."""
    idx = np.unique(np.round(np.arange(0.0, n, fps_src / fps_dst)).astype(int))
    return idx[idx < n]


def augment(xy, aspect, rng):
    """Random time and geometry changes of raw (T, 130, 2) landmarks; extract() runs afterwards."""
    a = AUG
    xy = xy.copy()
    if rng.random() < a["p_fps"] and len(xy) > 2:
        xy = xy[thin_index(len(xy), rng.uniform(*a["fps_range"]))]
    t = len(xy)
    if rng.random() < a["p_warp"] and t >= 4:
        steps = rng.uniform(*a["warp_speed"], size=t)
        pos = np.concatenate([[0.0], np.cumsum(steps[:-1])])
        pos = pos / pos[-1] * (t - 1)
        xy = xy[np.round(pos).astype(int)]
    if rng.random() < a["p_drop"] and t > 4:
        keep = rng.random(t) >= rng.uniform(0, a["drop_max"])
        keep[[0, -1]] = True
        xy = xy[keep]
    t = len(xy)
    if rng.random() < a["p_hand_gap"] and t >= 4:
        n = max(1, int(t * rng.uniform(*a["hand_gap"])))
        s = rng.integers(0, t - n + 1)
        xy[s:s + n, :42] = np.nan  # both hand blocks: Holistic lost the hands
    # geometry in isotropic units (y / aspect) around the shoulder centre
    xy[..., 1] /= aspect
    sh = xy[:, [F.L_SHOULDER, F.R_SHOULDER]].reshape(-1, 2)
    sh = sh[~np.isnan(sh).any(1)]
    c = sh.mean(0) if len(sh) else np.nanmean(xy.reshape(-1, 2), axis=0)
    p = xy - c
    if rng.random() < a["p_rotate"]:
        ang = np.deg2rad(rng.uniform(-a["rotate_deg"], a["rotate_deg"]))
        rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]], np.float32)
        p = p @ rot.T
    if rng.random() < a["p_stretch"]:
        p = p * rng.uniform(1 - a["stretch"], 1 + a["stretch"], size=2).astype(np.float32)
    xy = p + c
    xy[..., 1] *= aspect
    return xy


def _val_thinned_one(args):
    i, group, t = args
    xy = raw_clip(i)
    xy = xy[thin_index(len(xy), EVAL_FPS)]
    return F.extract(to_clip(xy), t_fix=t, aspect=F.dataset_aspect(group))


def val_thinned_features(data, t):
    args = [(int(i), int(data.groups[i]), t) for i in data.val_idx]
    with Pool(NUM_WORKERS) as pool:
        res = pool.map(_val_thinned_one, args, chunksize=64)
    return np.stack([r[0] for r in res]), np.stack([r[1] for r in res])


class OnlineDataset(torch.utils.data.Dataset):
    """Raw landmarks -> augment -> extract() in DataLoader workers."""

    def __init__(self, data, idx, t, train):
        self.groups, self.y = data.groups, data.y
        self.idx, self.t, self.train = idx, t, train

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        i = int(self.idx[k])
        xy = raw_clip(i)
        aspect = F.dataset_aspect(self.groups[i])
        if self.train:
            rng = np.random.default_rng((SEED, i, int(time.time_ns() % 2**31)))
            xy = augment(xy, aspect, rng)
        feats, mask = F.extract(to_clip(xy), t_fix=self.t, aspect=aspect)
        return torch.from_numpy(feats), torch.from_numpy(mask), int(self.y[i])


def tensor_batches(X, M, y, idx, batch, shuffle, rng):
    order = rng.permutation(idx) if shuffle else np.asarray(idx)
    for s in range(0, len(order), batch):
        b = np.sort(order[s:s + batch])
        yield (torch.from_numpy(np.asarray(X[b], dtype=np.float32)), torch.from_numpy(M[b]),
               torch.from_numpy(y[b]))


# --- Models ----------------------------------------------------------------------------------

class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))

    def forward(self, x):
        return (x - self.mean) / self.std


def safe_mask(mask):
    """At least one valid frame per clip, so pooling and softmax never divide by zero."""
    return mask | ~mask.any(dim=1, keepdim=True)


def masked_mean(h, mask):  # h (B, T, D), mask (B, T)
    m = mask.unsqueeze(-1).to(h.dtype)
    return (h * m).sum(1) / m.sum(1).clamp(min=1.0)


class CNN(nn.Module):
    def __init__(self, n_feat, n_cls, mean, std, width=192):
        super().__init__()
        self.norm = Normalize(mean, std)
        layers, c = [], n_feat
        for _ in range(3):
            layers += [nn.Conv1d(c, width, 5, padding=2), nn.BatchNorm1d(width), nn.ReLU(), nn.Dropout(0.1)]
            c = width
        self.net = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(2 * width, n_cls))

    def forward(self, x, mask):
        mask = safe_mask(mask)
        x = self.norm(x) * mask.unsqueeze(-1).to(x.dtype)
        h = self.net(x.transpose(1, 2)).transpose(1, 2)  # (B, T, W)
        mx = h.masked_fill(~mask.unsqueeze(-1), -1e4).amax(1)
        return self.head(torch.cat([masked_mean(h, mask), mx], dim=1))


class ConvBlock(nn.Module):
    def __init__(self, d, k=7, drop=0.1):
        super().__init__()
        self.dw = nn.Conv1d(d, d, k, padding=k // 2, groups=d)
        self.bn = nn.BatchNorm1d(d)
        self.pw = nn.Linear(d, d)
        self.drop = nn.Dropout(drop)

    def forward(self, x, mask):
        h = self.dw((x * mask.unsqueeze(-1).to(x.dtype)).transpose(1, 2))
        h = TF.silu(self.bn(h)).transpose(1, 2)
        return x + self.drop(self.pw(h))


class Attention(nn.Module):
    """Plain multi-head self-attention written out (stable ONNX export)."""

    def __init__(self, d, heads, drop):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.drop = nn.Dropout(drop)

    def forward(self, x, mask):
        b, t, d = x.shape
        q, k, v = self.qkv(x).reshape(b, t, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-1, -2)) / (self.dk ** 0.5)
        att = att.masked_fill(~mask[:, None, None, :], -1e4)
        att = self.drop(att.softmax(-1))
        return self.out((att @ v).transpose(1, 2).reshape(b, t, d))


class EncoderLayer(nn.Module):
    def __init__(self, d, heads, ff, drop):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = Attention(d, heads, drop)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(drop), nn.Linear(ff, d))
        self.drop = nn.Dropout(drop)

    def forward(self, x, mask):
        x = x + self.drop(self.att(self.ln1(x), mask))
        return x + self.drop(self.ff(self.ln2(x)))


class TransformerModel(nn.Module):
    def __init__(self, n_feat, n_cls, mean, std, t, d=256, heads=4, ff=512, layers=3, convs=2, drop=0.2):
        super().__init__()
        self.norm = Normalize(mean, std)
        self.inp = nn.Sequential(nn.Linear(n_feat, d), nn.LayerNorm(d))
        self.pos = nn.Parameter(torch.zeros(1, t, d))
        self.convs = nn.ModuleList([ConvBlock(d) for _ in range(convs)])
        self.layers = nn.ModuleList([EncoderLayer(d, heads, ff, drop) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(d, n_cls))

    def forward(self, x, mask):
        mask = safe_mask(mask)
        h = self.inp(self.norm(x)) + self.pos
        for blk in self.convs:
            h = blk(h, mask)
        for layer in self.layers:
            h = layer(h, mask)
        return self.head(masked_mean(self.ln(h), mask))


class WithSoftmax(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, features, mask):
        return self.model(features, mask).softmax(-1)


def build_model(kind, n_feat, n_cls, mean, std, t):
    if kind == "cnn":
        return CNN(n_feat, n_cls, mean, std)
    return TransformerModel(n_feat, n_cls, mean, std, t)


# --- Training --------------------------------------------------------------------------------

class GpuMonitor:
    """Samples nvidia-smi utilisation in a thread; mean over the last window."""

    def __init__(self):
        self.samples, self.stop = [], threading.Event()
        self.ok = shutil.which("nvidia-smi") is not None
        if self.ok:
            threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while not self.stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu",
                                      "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
                self.samples.append(float(out.stdout.strip().splitlines()[0]))
            except Exception:
                pass
            time.sleep(GPU_SAMPLE_S)

    def take(self):
        s, self.samples = self.samples, []
        return float(np.mean(s)) if s else None


def evaluate(model, batches):
    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for x, m, y in batches:
            with torch.autocast(DEVICE.type, enabled=DEVICE.type == "cuda"):
                logits = model(x.to(DEVICE, non_blocking=True), m.to(DEVICE, non_blocking=True))
            probs.append(logits.float().softmax(-1).cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(probs), np.concatenate(ys)


def topk(probs, y, k):
    return float((np.argsort(-probs, axis=1)[:, :k] == y[:, None]).any(1).mean())


def feature_stats(X, idx, rng, n=20000):
    sample = np.sort(rng.choice(idx, size=min(n, len(idx)), replace=False))
    x = np.asarray(X[sample], dtype=np.float32).reshape(-1, X.shape[-1])
    std = x.std(0)
    return x.mean(0), np.where(std < 1e-6, 1.0, std)


def train_one(data, cfg, t, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    X, M = data.features(t)
    if not cfg["augment"]:
        print(f"loading X{t} into RAM ...", flush=True)
        X = np.asarray(X)  # float16, ~6.5 GB for T=64
    mean, std = feature_stats(X, data.train_idx, rng)
    model = build_model(cfg["model"], X.shape[-1], data.n_classes, mean, std, t).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{out_dir.name}] model={cfg['model']} params={n_params:,} train={len(data.train_idx)} "
          f"val={len(data.val_idx)} val participants={data.val_participants}", flush=True)

    t0 = time.time()
    Xv13, Mv13 = val_thinned_features(data, t)
    print(f"val thinned to {EVAL_FPS} fps: {len(Xv13)} clips in {time.time() - t0:.0f}s", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    steps_per_epoch = int(np.ceil(len(data.train_idx) / cfg["batch"]))
    total, warm = cfg["epochs"] * steps_per_epoch, cfg["warmup_epochs"] * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, warm)) * 0.5 * (1 + np.cos(np.pi * min(1.0, s / max(1, total)))))
    scaler = torch.amp.GradScaler(enabled=DEVICE.type == "cuda")
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])

    if cfg["augment"]:
        loader = torch.utils.data.DataLoader(
            OnlineDataset(data, data.train_idx, t, train=True), batch_size=cfg["batch"], shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=4, drop_last=False)
    gpu = GpuMonitor()
    history, best, best_state, bad = [], -1.0, None, 0
    for epoch in range(cfg["epochs"]):
        model.train()
        batches = loader if cfg["augment"] else tensor_batches(X, M, data.y, data.train_idx, cfg["batch"], True, rng)
        te, wait, n, correct, loss_sum = time.time(), 0.0, 0, 0, 0.0
        gpu.take()
        tw = time.time()
        for x, m, y in batches:
            wait += time.time() - tw
            x, m, y = x.to(DEVICE, non_blocking=True), m.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            with torch.autocast(DEVICE.type, enabled=DEVICE.type == "cuda"):
                logits = model(x, m)
                loss = loss_fn(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            n += len(y)
            loss_sum += loss.item() * len(y)
            correct += int((logits.argmax(1) == y).sum())
            tw = time.time()
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        epoch_s = time.time() - te
        pv, yv = evaluate(model, tensor_batches(X, M, data.y, data.val_idx, 512, False, rng))
        row = {"epoch": epoch + 1, "train_loss": loss_sum / n, "train_acc": correct / n,
               "val_top1": topk(pv, yv, 1), "val_top3": topk(pv, yv, 3), "epoch_s": round(epoch_s, 1),
               "data_wait_share": round(wait / epoch_s, 3), "gpu_util_mean": gpu.take(),
               "lr": opt.param_groups[0]["lr"]}
        history.append(row)
        print(json.dumps(row), flush=True)
        if row["val_top1"] > best:
            best, bad = row["val_top1"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if cfg["patience"] and bad >= cfg["patience"]:
                print(f"early stop at epoch {epoch + 1}", flush=True)
                break
    gpu.stop.set()
    model.load_state_dict(best_state)

    pv, yv = evaluate(model, tensor_batches(X, M, data.y, data.val_idx, 512, False, rng))
    yv13 = data.y[data.val_idx]
    p13, _ = evaluate(model, tensor_batches(Xv13, Mv13, yv13, np.arange(len(Xv13)), 512, False, rng))
    results = {"orig": {"top1": topk(pv, yv, 1), "top3": topk(pv, yv, 3)},
               f"thinned_{EVAL_FPS:g}fps": {"top1": topk(p13, yv13, 1), "top3": topk(p13, yv13, 3)}}
    print(f"[{out_dir.name}] {json.dumps(results)}", flush=True)

    parity = export_onnx(model, t, X.shape[-1], Xv13[:ONNX_PARITY_SAMPLES], Mv13[:ONNX_PARITY_SAMPLES], out_dir)
    write_reports(data, pv, yv, p13, yv13, history, out_dir)
    metrics = {
        "preset": PRESET, "model": cfg["model"], "T_FIX": t, "F": int(X.shape[-1]),
        "FEATURES_VERSION": F.FEATURES_VERSION, "params": n_params, **results,
        "val_participants": data.val_participants, "n_train": int(len(data.train_idx)),
        "n_val": int(len(data.val_idx)), "split": f"GroupKFold({N_FOLDS}) by participant, fold {VAL_FOLD}",
        "config": cfg, "augmentation": AUG if cfg["augment"] else None, "eval_fps": EVAL_FPS,
        "onnx_parity_max_abs_diff": parity, "history": history,
        "versions": {"torch": torch.__version__, "device": str(DEVICE),
                     "gpu": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else None},
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
    (out_dir / "labels.json").write_text(json.dumps({str(k): v for k, v in sorted(data.signs.items())}, indent=1))
    return metrics


def export_onnx(model, t, n_feat, xs, ms, out_dir):
    """Export model + softmax to ONNX and check it against PyTorch on real validation samples."""
    import onnxruntime as ort

    model = WithSoftmax(model).cpu().float().eval()
    path = out_dir / "model.onnx"
    dummy = (torch.zeros(1, t, n_feat), torch.ones(1, t, dtype=torch.bool))
    kwargs = dict(input_names=["features", "mask"], output_names=["probs"], opset_version=ONNX_OPSET,
                  dynamic_axes={"features": {0: "batch"}, "mask": {0: "batch"}, "probs": {0: "batch"}})
    try:
        torch.onnx.export(model, dummy, str(path), dynamo=False, **kwargs)
    except TypeError:  # older torch without the `dynamo` argument
        torch.onnx.export(model, dummy, str(path), **kwargs)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x, m = np.asarray(xs, np.float32), np.asarray(ms, bool)
    with torch.no_grad():
        ref = model(torch.from_numpy(x), torch.from_numpy(m)).numpy()
    got = sess.run(["probs"], {"features": x, "mask": m})[0]
    diff = float(np.abs(ref - got).max())
    print(f"ONNX parity on {len(x)} samples: max abs diff {diff:.2e}", flush=True)
    if diff > ONNX_PARITY_ATOL:
        raise RuntimeError(f"ONNX parity failed: {diff:.2e} > {ONNX_PARITY_ATOL}")
    return diff


def write_reports(data, pv, yv, p13, yv13, history, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = data.n_classes
    pred = pv.argmax(1)
    cm = np.zeros((n, n), np.int64)
    np.add.at(cm, (yv, pred), 1)
    acc13 = (p13.argmax(1) == yv13)
    per = pd.DataFrame({"class_id": np.arange(n), "sign": [data.signs[i] for i in range(n)],
                        "n_val": cm.sum(1), "accuracy": np.diag(cm) / np.maximum(cm.sum(1), 1),
                        f"accuracy_{EVAL_FPS:g}fps": [acc13[yv13 == i].mean() if (yv13 == i).any() else np.nan
                                                      for i in range(n)]})
    per.sort_values("accuracy").to_csv(out_dir / "per_sign_accuracy.csv", index=False)
    off = cm.copy()
    np.fill_diagonal(off, 0)
    top = np.dstack(np.unravel_index(np.argsort(-off, axis=None)[:20], off.shape))[0]
    pd.DataFrame([{"true": data.signs[a], "predicted": data.signs[b], "count": int(off[a, b]),
                   "share_of_true": round(off[a, b] / max(cm[a].sum(), 1), 3)} for a, b in top]
                 ).to_csv(out_dir / "top_confusions.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 10), dpi=120)
    ax.imshow(cm / np.maximum(cm.sum(1, keepdims=True), 1), cmap="viridis", vmin=0, vmax=1)
    ax.set_title("Confusion matrix (rows: true sign, normalised)")
    ax.set_xlabel("predicted class id")
    ax.set_ylabel("true class id")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png")
    plt.close(fig)

    h = pd.DataFrame(history)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4), dpi=120)
    a1.plot(h.epoch, h.train_loss, label="train loss")
    a1.set_xlabel("epoch")
    a1.legend()
    a2.plot(h.epoch, h.train_acc, label="train top-1")
    a2.plot(h.epoch, h.val_top1, label="val top-1")
    a2.plot(h.epoch, h.val_top3, label="val top-3")
    a2.set_xlabel("epoch")
    a2.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png")
    plt.close(fig)


def main():
    cfg = PRESETS[PRESET]
    print(f"preset={PRESET} cfg={cfg} device={DEVICE} torch={torch.__version__} "
          f"FEATURES_VERSION={F.FEATURES_VERSION}", flush=True)
    data = Data()
    summary = {}
    for t in cfg["t_values"]:
        m = train_one(data, cfg, t, OUT_ROOT / f"{PRESET}_T{t}")
        summary[f"T{t}"] = {k: m[k] for k in m if k.startswith(("orig", "thinned"))}
        summary[f"T{t}"]["epochs_run"] = len(m["history"])
        summary[f"T{t}"]["mean_epoch_s"] = float(np.mean([r["epoch_s"] for r in m["history"]]))
        summary[f"T{t}"]["gpu_util_mean"] = m["history"][-1]["gpu_util_mean"]
        summary[f"T{t}"]["data_wait_share"] = float(np.mean([r["data_wait_share"] for r in m["history"]]))
    (OUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
