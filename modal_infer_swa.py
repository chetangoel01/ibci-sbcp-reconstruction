"""Re-run inference on existing SWA checkpoint (no Hann, equal-weight overlap).

Usage:
    modal run --detach modal_infer_swa.py
"""

import modal

app = modal.App("phase2-infer-swa")

data_vol = modal.Volume.from_name("phase2-data", create_if_missing=True)
output_vol = modal.Volume.from_name("phase2-outputs-gru-swa", create_if_missing=True)

base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "pandas", "scipy", "tqdm")
)

code_image = (
    base_image
    .add_local_file("phase2_config.py", "/root/repo/phase2_config.py")
    .add_local_file("phase2_data.py", "/root/repo/phase2_data.py")
    .add_local_file("phase2_model.py", "/root/repo/phase2_model.py")
    .add_local_file("phase2_inference.py", "/root/repo/phase2_inference.py")
)


@app.function(
    image=code_image,
    gpu="A10G",
    timeout=3600,
    volumes={"/root/data": data_vol, "/root/outputs": output_vol},
)
def run_infer():
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
        context_bins=400,
        seed=44,
        gru_d_model=128,
        gru_n_layers=3,
        gru_dropout=0.2,
    )

    for sigma in [0, 3]:
        print(f"Inference with sigma={sigma}...")
        infer_fn(config, smooth_sigma=float(sigma), tag="swa")
    output_vol.commit()

    print("\nDone! Download with:")
    print("  modal volume get phase2-outputs-gru-swa results/ .")


@app.local_entrypoint()
def main():
    run_infer.remote()
