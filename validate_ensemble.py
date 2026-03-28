"""Validate all models and find the best ensemble combination.

Runs each model on the validation set, then tries every possible combination
of models (equal weights + optimized weights) to find the best ensemble.
"""

import itertools
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize

from phase2_config import get_config, set_global_seeds
from phase2_data import (
    SessionCache, discover_session_ids, split_train_val,
    load_sbp, zscore_session, get_active_mask,
)
from phase2_model import GRUDecoder, TransformerDecoder, POYODecoder

SEED = 42
VAL_SESSIONS = 15

MODELS = {
    "gru_ctx800": {
        "checkpoint": "phase2_outputs/checkpoints/gru_ctx800/best_gru.pt",
        "submission": "submission_gru_ctx800_seed44_sigma3.csv",
        "kaggle_r2": 0.6705,
    },
    "gru_ctx600": {
        "checkpoint": "phase2_outputs/checkpoints/gru_ctx600/best_gru.pt",
        "submission": "submission_gru_ctx600_seed44_sigma3.csv",
        "kaggle_r2": 0.6658,
    },
    "gru_ctx400": {
        "checkpoint": "phase2_outputs/checkpoints/gru_ctx400/best_gru.pt",
        "submission": "submission_gru_ctx400_seed44_sigma3.csv",
        "kaggle_r2": 0.6555,
    },
    "gru_wide": {
        "checkpoint": "phase2_outputs/checkpoints/gru_wide/best_gru.pt",
        "submission": None,
        "kaggle_r2": None,
    },
    "gru_ctx200": {
        "checkpoint": "phase2_outputs/checkpoints/gru_default/best_gru.pt",
        "submission": "submission_gru_ctx200_seed44_sigma3.csv",
        "kaggle_r2": None,
    },
    "transformer": {
        "checkpoint": "phase2_outputs/checkpoints/best_transformer.pt",
        "submission": "submission_transformer.csv",
        "kaggle_r2": None,
    },
    "poyo": {
        "checkpoint": "phase2_outputs/checkpoints/best_poyo.pt",
        "submission": "phase2_outputs/results/submission_poyo.csv",
        "kaggle_r2": None,
    },
}


def compute_r2(pred, true):
    ss_res = np.sum((pred - true) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    if ss_tot < 1e-10:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def load_model_from_checkpoint(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model_type = cfg["model_type"]
    ctx = cfg["context_bins"]

    if model_type == "gru":
        model = GRUDecoder(
            n_channels=96,
            d_model=cfg["gru_d_model"],
            n_layers=cfg["gru_n_layers"],
            dropout=cfg["gru_dropout"],
            n_outputs=2,
        )
    elif model_type == "transformer":
        model = TransformerDecoder(
            n_channels=96,
            d_model=cfg["tf_d_model"],
            n_layers=cfg["tf_n_layers"],
            n_heads=cfg["tf_n_heads"],
            dim_ff=cfg["tf_dim_ff"],
            dropout=cfg["tf_dropout"],
            max_context=ctx * 2,
            n_outputs=2,
        )
    elif model_type == "poyo":
        model = POYODecoder(
            n_channels=96,
            d_model=cfg["poyo_d_model"],
            n_latents=cfg["poyo_n_latents"],
            n_self_attn_layers=cfg["poyo_n_self_attn_layers"],
            n_heads=cfg["poyo_n_heads"],
            dim_ff=cfg["poyo_dim_ff"],
            dropout=cfg["poyo_dropout"],
            max_context=ctx * 2,
            n_outputs=2,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.train(False)
    return model, ctx, model_type


def predict_session_val(model, sbp_np, active_np, ctx, device):
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
        mask = torch.from_numpy(active_np).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(inp, active_mask=mask)
        return out.squeeze(0).cpu().numpy()[:n_bins].astype(np.float32)

    mask_t = torch.from_numpy(active_np).to(device)

    for batch_start in range(0, len(starts), 32):
        batch_idx = starts[batch_start:batch_start + 32]
        batch_inp = np.stack([sbp_np[s:s + ctx] for s in batch_idx])
        inp = torch.from_numpy(batch_inp).to(device)
        mask_b = mask_t.unsqueeze(0).expand(len(batch_idx), -1)

        with torch.no_grad():
            out = model(inp, active_mask=mask_b)
        out_np = out.cpu().numpy()

        for j, s in enumerate(batch_idx):
            preds[s:s + ctx] += out_np[j]
            counts[s:s + ctx] += 1.0

    counts = np.maximum(counts, 1.0)
    preds /= counts[:, None]
    return preds.astype(np.float32)


def compute_ensemble_r2(model_preds, weights, val_ids, val_cache, model_names):
    w = np.array(weights)
    w = w / w.sum()

    all_r2 = []
    for sid in val_ids:
        kin = val_cache.kinematics[sid][:, :2]
        n_bins = kin.shape[0]

        ens_pred = np.zeros((n_bins, 2), dtype=np.float64)
        for i, name in enumerate(model_names):
            ens_pred += w[i] * model_preds[name][sid]

        for c in range(2):
            all_r2.append(compute_r2(ens_pred[:, c], kin[:, c]))

    return float(np.mean(all_r2))


def optimize_weights(model_preds, val_ids, val_cache, model_names):
    n = len(model_names)
    if n == 1:
        return [1.0]

    def neg_r2(log_weights):
        w = np.exp(log_weights)
        w = w / w.sum()
        return -compute_ensemble_r2(model_preds, w, val_ids, val_cache, model_names)

    best_result = None
    best_val = float("inf")

    for _ in range(10):
        x0 = np.random.randn(n) * 0.5
        result = minimize(neg_r2, x0, method="Nelder-Mead",
                          options={"maxiter": 500, "xatol": 1e-6, "fatol": 1e-8})
        if result.fun < best_val:
            best_val = result.fun
            best_result = result

    w = np.exp(best_result.x)
    w = w / w.sum()
    return w.tolist()


def main():
    set_global_seeds(SEED)
    config = get_config("local")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    all_train_ids = discover_session_ids(config.train_dir)
    train_ids, val_ids = split_train_val(all_train_ids, n_val=VAL_SESSIONS, seed=SEED)
    print(f"Validation sessions: {len(val_ids)}")

    print("Loading validation cache...")
    val_cache = SessionCache(config.train_dir, val_ids, has_kinematics=True)

    model_preds = {}
    model_val_r2 = {}

    for name, info in MODELS.items():
        ckpt_path = Path(info["checkpoint"])
        if not ckpt_path.exists():
            print(f"  SKIP {name}: checkpoint not found at {ckpt_path}")
            continue

        print(f"\n{'='*60}")
        print(f"Validating: {name}")
        t0 = time.time()

        model, ctx, mtype = load_model_from_checkpoint(ckpt_path, device)
        print(f"  Type: {mtype}, ctx={ctx}, params={sum(p.numel() for p in model.parameters()):,}")

        preds = {}
        all_r2 = []
        for sid in val_ids:
            sbp_np = val_cache.sbp[sid]
            active_np = val_cache.active_masks[sid]
            kin_np = val_cache.kinematics[sid][:, :2]

            pred = predict_session_val(model, sbp_np, active_np, ctx, device)
            preds[sid] = pred

            for c in range(2):
                n = min(pred.shape[0], kin_np.shape[0])
                all_r2.append(compute_r2(pred[:n, c], kin_np[:n, c]))

        mean_r2 = float(np.mean(all_r2))
        model_preds[name] = preds
        model_val_r2[name] = mean_r2

        elapsed = time.time() - t0
        kaggle = info.get("kaggle_r2")
        kaggle_str = f"  Kaggle={kaggle:.4f}" if kaggle else ""
        print(f"  Val R2={mean_r2:.4f}{kaggle_str}  ({elapsed:.1f}s)")

        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    print(f"\n{'='*60}")
    print("INDIVIDUAL MODEL VALIDATION R2:")
    print(f"{'='*60}")
    sorted_models = sorted(model_val_r2.items(), key=lambda x: -x[1])
    for name, r2 in sorted_models:
        kaggle = MODELS[name].get("kaggle_r2")
        k_str = f"  (Kaggle={kaggle:.4f})" if kaggle else ""
        print(f"  {name:20s}  val_R2={r2:.4f}{k_str}")

    available = list(model_preds.keys())
    print(f"\n{'='*60}")
    print(f"ENSEMBLE COMBINATIONS ({len(available)} models, {2**len(available)-1} combos):")
    print(f"{'='*60}")

    results = []
    for r in range(2, len(available) + 1):
        for combo in itertools.combinations(available, r):
            combo_names = list(combo)

            eq_r2 = compute_ensemble_r2(
                model_preds, [1.0] * len(combo_names),
                val_ids, val_cache, combo_names
            )

            opt_w = optimize_weights(model_preds, val_ids, val_cache, combo_names)
            opt_r2 = compute_ensemble_r2(
                model_preds, opt_w, val_ids, val_cache, combo_names
            )

            results.append({
                "models": combo_names,
                "equal_r2": eq_r2,
                "opt_r2": opt_r2,
                "opt_weights": opt_w,
            })

    results.sort(key=lambda x: -x["opt_r2"])

    print(f"\nTop 20 ensemble combinations:")
    print(f"{'-'*80}")
    for i, res in enumerate(results[:20]):
        names = " + ".join(res["models"])
        w_str = ", ".join(f"{w:.3f}" for w in res["opt_weights"])
        print(f"  {i+1:2d}. eq={res['equal_r2']:.4f}  opt={res['opt_r2']:.4f}  [{w_str}]")
        print(f"      {names}")

    best_single = sorted_models[0]
    print(f"\n  Best single: {best_single[0]} = {best_single[1]:.4f}")
    print(f"  Best ensemble: opt_R2={results[0]['opt_r2']:.4f} ({' + '.join(results[0]['models'])})")

    print(f"\nCombinations that beat best single model ({best_single[1]:.4f}):")
    for res in results:
        if res["opt_r2"] > best_single[1]:
            names = " + ".join(res["models"])
            w_str = ", ".join(f"{w:.3f}" for w in res["opt_weights"])
            print(f"  opt={res['opt_r2']:.4f}  [{w_str}]  {names}")
        else:
            break


if __name__ == "__main__":
    main()
