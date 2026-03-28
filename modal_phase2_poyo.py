"""Train POYO model on Modal with A10G GPU.

Usage:
    # First time: upload data to a Modal volume
    modal run modal_phase2_poyo.py::upload_data

    # Then train + infer
    modal run modal_phase2_poyo.py
"""

import modal

app = modal.App("phase2-poyo-training")

data_vol = modal.Volume.from_name("phase2-data", create_if_missing=True)
output_vol = modal.Volume.from_name("phase2-outputs", create_if_missing=True)

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


@app.function(image=base_image, volumes={"/root/data": data_vol}, timeout=1800)
def upload_data():
    """Upload local data to a Modal volume (run once)."""
    import os
    from pathlib import Path

    data_dir = Path("/root/data")
    print(f"Checking volume at {data_dir}...")
    if (data_dir / "train").exists() and len(list((data_dir / "train").glob("*.npy"))) > 100:
        print("Data already uploaded. Skipping.")
        return

    print("Volume is empty — data will be uploaded via local entrypoint.")


@app.function(
    image=code_image,
    gpu="A10G",
    timeout=10800,
    volumes={"/root/data": data_vol, "/root/outputs": output_vol},
)
def train_poyo():
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
        model_type="poyo",
        context_bins=50,
        batch_size=32,
        lr=3e-4,
        epochs=60,
        warmup_epochs=5,
        val_sessions=15,
        num_workers=4,
        poyo_d_model=128,
        poyo_n_latents=64,
        poyo_n_self_attn_layers=6,
        poyo_n_heads=8,
        poyo_dim_ff=512,
        poyo_dropout=0.1,
    )

    train_fn(config)
    output_vol.commit()
    print("POYO training complete.")


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
        model_type="poyo",
        context_bins=50,
        poyo_d_model=128,
        poyo_n_latents=64,
        poyo_n_self_attn_layers=6,
        poyo_n_heads=8,
        poyo_dim_ff=512,
        poyo_dropout=0.1,
    )

    infer_fn(config, smooth_sigma=1.5)
    output_vol.commit()
    print("POYO inference complete.")


@app.local_entrypoint()
def main():
    import os
    import subprocess
    from pathlib import Path

    local_data = Path("phase2_v2_kaggle_data")

    # Upload data to volume if needed
    print("Checking if data volume needs uploading...")
    result = subprocess.run(
        ["modal", "volume", "ls", "phase2-data", "train/"],
        capture_output=True, text=True
    )
    if "D001_sbp.npy" not in result.stdout:
        print("Uploading data to Modal volume (this may take a few minutes)...")
        subprocess.run(
            ["modal", "volume", "put", "phase2-data", str(local_data / "train") + "/", "train/"],
            check=True,
        )
        subprocess.run(
            ["modal", "volume", "put", "phase2-data", str(local_data / "test") + "/", "test/"],
            check=True,
        )
        subprocess.run(
            ["modal", "volume", "put", "phase2-data", str(local_data / "sample_submission.csv"), "sample_submission.csv", "--force"],
            check=True,
        )
        subprocess.run(
            ["modal", "volume", "put", "phase2-data", str(local_data / "test_index.csv"), "test_index.csv", "--force"],
            check=True,
        )
        print("Data upload complete.")
    else:
        print("Data already in volume, skipping upload.")

    print("Starting POYO training on A10G...")
    train_poyo.remote()
    print("Training done. Running inference...")
    run_inference.remote()
    print("All done!")
    print("Download results:")
    print("  modal volume get phase2-outputs checkpoints/best_poyo.pt .")
    print("  modal volume get phase2-outputs results/submission_poyo.csv .")
