"""Train bidirectional GRU on Modal with A10G GPU.

Provides ensemble diversity alongside the Transformer models.
GRU has a different inductive bias (recurrent vs attention) —
ensembling with transformers typically yields 0.01-0.03 leaderboard gain.

Usage:
    modal run modal_train_gru.py
"""

import modal

app = modal.App("phase2-gru")

data_vol = modal.Volume.from_name("phase2-data", create_if_missing=True)
output_vol = modal.Volume.from_name("phase2-outputs-gru", create_if_missing=True)

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


@app.function(
    image=code_image,
    gpu="A10G",
    timeout=10800,
    volumes={"/root/data": data_vol, "/root/outputs": output_vol},
)
def train_gru():
    import sys
    from pathlib import Path

    sys.path.insert(0, "/root/repo")

    from phase2_config import Phase2Config
    from phase2_train import train as train_fn

    data_vol.reload()

    train_dir = Path("/root/data/train")
    n_files = len(list(train_dir.glob("*.npy")))
    print(f"Data volume has {n_files} npy files in train/")

    config = Phase2Config(
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
        seed=44,
        gru_d_model=128,
        gru_n_layers=3,
        gru_dropout=0.2,
    )

    train_fn(config)
    output_vol.commit()
    print("GRU training complete.")


@app.function(
    image=code_image,
    gpu="A10G",
    timeout=3600,
    volumes={"/root/data": data_vol, "/root/outputs": output_vol},
)
def run_inference():
    import sys
    from pathlib import Path

    sys.path.insert(0, "/root/repo")

    from phase2_config import Phase2Config
    from phase2_inference import run_inference as infer_fn

    data_vol.reload()
    output_vol.reload()

    config = Phase2Config(
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
        gru_d_model=128,
        gru_n_layers=3,
        gru_dropout=0.2,
    )

    infer_fn(config, smooth_sigma=10.0)
    output_vol.commit()
    print("Inference complete.")


@app.local_entrypoint()
def main():
    print("Starting GRU training on A10G (augmentation + velocity loss)...")
    train_gru.remote()
    print("Training done. Running inference...")
    run_inference.remote()
    print("All done!")
    print("Download results:")
    print("  modal volume get phase2-outputs-gru checkpoints/best_gru.pt .")
    print("  modal volume get phase2-outputs-gru results/submission_gru.csv .")
