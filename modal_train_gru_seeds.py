"""Train GRU ctx=200 with two new seeds on Modal A10G.

Runs seed=45 and seed=46 sequentially, generates submissions for each.

Usage:
    modal run modal_train_gru_seeds.py
"""

import modal

app = modal.App("phase2-gru-seeds")

data_vol = modal.Volume.from_name("phase2-data", create_if_missing=True)
output_vol = modal.Volume.from_name("phase2-outputs-gru-seeds", create_if_missing=True)

base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "pandas", "scipy", "tqdm")
)

code_image = (
    base_image
    .add_local_file("phase2_config.py", "/root/repo/phase2_config.py")
    .add_local_file("phase2_data.py", "/root/repo/phase2_data.py")
    .add_local_file("phase2_model.py", "/root/repo/phase2_model.py")
    .add_local_file("phase2_train.py", "/root/repo/phase2_train.py")
    .add_local_file("phase2_inference.py", "/root/repo/phase2_inference.py")
)


def make_config(seed):
    from pathlib import Path
    from phase2_config import Phase2Config

    return Phase2Config(
        profile="modal",
        repo_root=Path("/root/repo"),
        data_dir=Path("/root/data"),
        train_dir=Path("/root/data/train"),
        test_dir=Path("/root/data/test"),
        output_dir=Path("/root/outputs"),
        checkpoints_dir=Path("/root/outputs/checkpoints"),
        results_dir=Path("/root/outputs/results"),
        logs_dir=Path("/root/outputs/logs"),
        metadata_path=Path("/root/data/metadata.csv"),
        sample_sub_path=Path("/root/data/sample_submission.csv"),
        test_index_path=Path("/root/data/test_index.csv"),
        device="cuda",
        model_type="gru",
        context_bins=200,
        batch_size=64,
        lr=3e-4,
        epochs=80,
        warmup_epochs=5,
        val_sessions=15,
        num_workers=4,
        velocity_aux_weight=0.1,
        seed=seed,
        gru_d_model=128,
        gru_n_layers=3,
        gru_dropout=0.2,
    )


@app.function(
    image=code_image,
    gpu="A10G",
    timeout=10800,
    volumes={"/root/data": data_vol, "/root/outputs": output_vol},
)
def train_seed(seed: int):
    import shutil
    import sys
    sys.path.insert(0, "/root/repo")

    from pathlib import Path
    from phase2_train import train as train_fn
    from phase2_inference import run_inference as infer_fn

    data_vol.reload()
    output_vol.reload()

    config = make_config(seed)
    print(f"Training GRU ctx=200 seed={seed}")

    train_fn(config)

    # Rename checkpoint to include seed so parallel runs don't clobber
    src = config.checkpoints_dir / "best_gru.pt"
    dst = config.checkpoints_dir / f"best_gru_seed{seed}.pt"
    shutil.copy2(src, dst)
    output_vol.commit()

    # Inference with sigma=3 (best from sweep)
    print(f"\nInference with sigma=3...")
    infer_fn(config, smooth_sigma=3.0)
    output_vol.commit()

    # Read back the best val R2 from checkpoint
    import torch
    ckpt = torch.load(dst, map_location="cpu", weights_only=False)
    val_r2 = ckpt.get("val_r2", 0)
    print(f"\nSeed {seed} done. Best val R2: {val_r2:.4f}")
    return {"seed": seed, "val_r2": val_r2}


@app.local_entrypoint()
def main():
    # Run both seeds in parallel on separate GPUs
    results = []
    for result in train_seed.map([45, 46]):
        results.append(result)
        print(f"Seed {result['seed']}: val_R2={result['val_r2']:.4f}")

    best = max(results, key=lambda r: r["val_r2"])
    print(f"\nBest seed: {best['seed']} (val_R2={best['val_r2']:.4f})")
    print("Download with:")
    print("  modal volume get phase2-outputs-gru-seeds results/ .")
