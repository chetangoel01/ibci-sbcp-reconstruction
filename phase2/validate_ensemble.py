"""Fast ensemble validation: focus on top GRU models only."""

import itertools
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize

from config import get_config, set_global_seeds
from data import SessionCache, discover_session_ids, split_train_val
from model import GRUDecoder

SEED = 44  # must match training seed so val sessions are truly held-out
VAL_SESSIONS = 15

MODELS = {
    "gru_wide": {
        "checkpoint": "outputs/checkpoints/gru_wide/best_gru.pt",
        "kaggle_r2": None,
    },
    "gru_ctx800": {
        "checkpoint": "outputs/checkpoints/gru_ctx800/best_gru.pt",
        "kaggle_r2": 0.6705,
    },
    "gru_ctx600": {
        "checkpoint": "outputs/checkpoints/gru_ctx600/best_gru.pt",
        "kaggle_r2": 0.6658,
    },
}


def compute_r2(pred, true):
    ss_res = np.sum((pred - true) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    if ss_tot < 1e-10:
        return 0.0
    return 1.0 - ss_res / ss_tot


def predict_session(model, sbp_np, active_np, ctx, device):
    n_bins = sbp_np.shape[0]
    preds = np.zeros((n_bins, 2), dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.float64)

    stride = max(1, ctx // 2)
    starts = list(range(0, n_bins - ctx + 1, stride))
    if starts and (starts[-1] + ctx) < n_bins:
        starts.append(n_bins - ctx)

    if not starts:
        pad_len = ctx - n_bins
        padded = np.pad(sbp_np, ((0, pad_len), (0, 0)), mode="edge")
        inp = torch.from_numpy(padded).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(inp)
        return out.squeeze(0).cpu().numpy()[:n_bins].astype(np.float32)

    for batch_start in range(0, len(starts), 32):
        batch_idx = starts[batch_start:batch_start + 32]
        batch_inp = np.stack([sbp_np[s:s + ctx] for s in batch_idx])
        inp = torch.from_numpy(batch_inp).to(device)

        with torch.no_grad():
            out = model(inp)
        out_np = out.cpu().numpy()

        for j, s in enumerate(batch_idx):
            preds[s:s + ctx] += out_np[j]
            counts[s:s + ctx] += 1.0

    counts = np.maximum(counts, 1.0)
    preds /= counts[:, None]
    return preds.astype(np.float32)


def ensemble_r2(model_preds, weights, val_ids, val_cache, names):
    w = np.array(weights, dtype=np.float64)
    w /= w.sum()

    all_r2 = []
    for sid in val_ids:
        kin = val_cache.kinematics[sid][:, :2]
        n = kin.shape[0]
        ens = np.zeros((n, 2), dtype=np.float64)
        for i, nm in enumerate(names):
            ens += w[i] * model_preds[nm][sid][:n]
        for c in range(2):
            all_r2.append(compute_r2(ens[:, c], kin[:, c]))
    return float(np.mean(all_r2))


def optimize_weights(model_preds, val_ids, val_cache, names):
    n = len(names)
    if n == 1:
        return [1.0], ensemble_r2(model_preds, [1.0], val_ids, val_cache, names)

    def neg_r2(log_w):
        w = np.exp(log_w)
        w /= w.sum()
        return -ensemble_r2(model_preds, w, val_ids, val_cache, names)

    best_r = None
    best_v = float("inf")
    for _ in range(5):
        x0 = np.random.randn(n) * 0.3
        r = minimize(neg_r2, x0, method="Nelder-Mead",
                     options={"maxiter": 300, "xatol": 1e-5, "fatol": 1e-7})
        if r.fun < best_v:
            best_v = r.fun
            best_r = r

    w = np.exp(best_r.x)
    w /= w.sum()
    return w.tolist(), -best_v


def main():
    set_global_seeds(SEED)
    config = get_config("local")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    all_train_ids = discover_session_ids(config.train_dir)
    _, val_ids = split_train_val(all_train_ids, n_val=VAL_SESSIONS, seed=SEED)
    print(f"Validation: {len(val_ids)} sessions")
    print("Loading val cache...")
    val_cache = SessionCache(config.train_dir, val_ids, has_kinematics=True)

    # Run each model
    model_preds = {}
    print()
    for name, info in MODELS.items():
        ckpt = torch.load(info["checkpoint"], map_location=device, weights_only=False)
        cfg = ckpt["config"]
        ctx = cfg["context_bins"]
        model = GRUDecoder(
            n_channels=96, d_model=cfg["gru_d_model"],
            n_layers=cfg["gru_n_layers"], dropout=cfg["gru_dropout"], n_outputs=2,
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.train(False)

        t0 = time.time()
        preds = {}
        all_r2 = []
        for sid in val_ids:
            pred = predict_session(model, val_cache.sbp[sid], val_cache.active_masks[sid], ctx, device)
            preds[sid] = pred
            kin = val_cache.kinematics[sid][:, :2]
            n = min(pred.shape[0], kin.shape[0])
            for c in range(2):
                all_r2.append(compute_r2(pred[:n, c], kin[:n, c]))

        r2 = float(np.mean(all_r2))
        model_preds[name] = preds
        k = info.get("kaggle_r2")
        k_str = f" (Kaggle={k:.4f})" if k else ""
        print(f"{name:15s} val_R2={r2:.4f}{k_str}  d={cfg['gru_d_model']} ctx={ctx} ({time.time()-t0:.1f}s)")
        del model

    # All combos of 2+ models
    avail = list(model_preds.keys())
    print(f"\nTesting {2**len(avail) - len(avail) - 1} ensemble combinations...")
    t0 = time.time()

    results = []
    for r in range(2, len(avail) + 1):
        for combo in itertools.combinations(avail, r):
            names = list(combo)
            eq = ensemble_r2(model_preds, [1.0]*len(names), val_ids, val_cache, names)
            opt_w, opt = optimize_weights(model_preds, val_ids, val_cache, names)
            results.append({"models": names, "eq": eq, "opt": opt, "weights": opt_w})

    results.sort(key=lambda x: -x["opt"])
    print(f"Done ({time.time()-t0:.1f}s)\n")

    print("=" * 80)
    print("TOP 25 ENSEMBLE COMBINATIONS")
    print("=" * 80)
    for i, res in enumerate(results[:25]):
        w_str = ", ".join(f"{w:.3f}" for w in res["weights"])
        models = " + ".join(res["models"])
        print(f"{i+1:2d}. eq={res['eq']:.4f}  opt={res['opt']:.4f}  [{w_str}]")
        print(f"    {models}")

    # Best single
    best_single = max(model_preds.keys(), key=lambda n: ensemble_r2(model_preds, [1.0], val_ids, val_cache, [n]))
    best_single_r2 = ensemble_r2(model_preds, [1.0], val_ids, val_cache, [best_single])
    print(f"\nBest single: {best_single} = {best_single_r2:.4f}")
    print(f"Best ensemble: {results[0]['opt']:.4f} ({' + '.join(results[0]['models'])})")

    beaten = [r for r in results if r["opt"] > best_single_r2]
    if beaten:
        print(f"\n{len(beaten)} combos beat best single:")
        for r in beaten:
            w_str = ", ".join(f"{w:.3f}" for w in r["weights"])
            print(f"  opt={r['opt']:.4f}  [{w_str}]  {' + '.join(r['models'])}")


if __name__ == "__main__":
    main()
