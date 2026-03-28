"""Download kinematics from LINK dataset (DANDI #001201) for Phase 2 test sessions.

Matches DANDI session dates to Kaggle D### IDs chronologically,
then downloads NWB files and extracts kinematics for the 125 test sessions.
"""

import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import requests

DATA_DIR = Path(__file__).parent / "phase2_v2_kaggle_data"
TEST_DIR = DATA_DIR / "test"
LINK_DIR = Path(__file__).parent / "link_nwb"
LINK_DIR.mkdir(exist_ok=True)

DANDI_API = "https://api.dandiarchive.org/api"
DANDISET_ID = "001201"
VERSION = "draft"


def get_dandi_assets():
    """Fetch all asset metadata from DANDI, sorted by path (chronological)."""
    assets = []
    url = f"{DANDI_API}/dandisets/{DANDISET_ID}/versions/{VERSION}/assets/"
    params = {"page_size": 200, "order": "path"}

    while url:
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        assets.extend(data["results"])
        url = data.get("next")
        params = {}  # next URL already has params

    # Sort by path to ensure chronological order
    assets.sort(key=lambda a: a["path"])
    return assets


def build_d_mapping(assets):
    """Map D### IDs to DANDI assets (chronological ordering)."""
    mapping = {}
    for i, asset in enumerate(assets):
        did = f"D{i + 1:03d}"
        mapping[did] = {
            "asset_id": asset["asset_id"],
            "path": asset["path"],
            "size": asset.get("size", 0),
        }
    return mapping


def download_nwb(asset_id, out_path):
    """Download a single NWB file from DANDI."""
    url = f"{DANDI_API}/assets/{asset_id}/download/"
    resp = requests.get(url, stream=True, allow_redirects=True)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0

    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = downloaded / total * 100
                print(f"\r  {downloaded / 1e6:.1f}/{total / 1e6:.1f} MB ({pct:.0f}%)", end="", flush=True)
    print()


def extract_kinematics_from_nwb(nwb_path):
    """Extract kinematics from an NWB file using h5py directly."""
    with h5py.File(nwb_path, "r") as f:
        analysis = f["analysis"]

        # Try different field names
        idx_key = None
        mrs_key = None
        for key in analysis.keys():
            kl = key.lower()
            if "index" in kl and "pos" in kl:
                idx_key = key
            elif ("mrs" in kl or "mrp" in kl or "middle" in kl) and "pos" in kl:
                mrs_key = key

        if idx_key is None or mrs_key is None:
            # Fallback: try common names
            for key in analysis.keys():
                kl = key.lower()
                if "index" in kl and "vel" not in kl:
                    idx_key = idx_key or key
                elif ("mrs" in kl or "mrp" in kl) and "vel" not in kl:
                    mrs_key = mrs_key or key

        if idx_key is None or mrs_key is None:
            print(f"  WARNING: Could not find position keys. Available: {list(analysis.keys())}")
            return None

        idx_pos = analysis[idx_key]["data"][:].ravel().astype(np.float32)
        mrs_pos = analysis[mrs_key]["data"][:].ravel().astype(np.float32)

        # Also get velocities if available
        idx_vel_key = None
        mrs_vel_key = None
        for key in analysis.keys():
            kl = key.lower()
            if "index" in kl and "vel" in kl:
                idx_vel_key = key
            elif ("mrs" in kl or "mrp" in kl) and "vel" in kl:
                mrs_vel_key = key

        if idx_vel_key and mrs_vel_key:
            idx_vel = analysis[idx_vel_key]["data"][:].ravel().astype(np.float32)
            mrs_vel = analysis[mrs_vel_key]["data"][:].ravel().astype(np.float32)
            kin = np.column_stack([idx_pos, mrs_pos, idx_vel, mrs_vel])
        else:
            kin = np.column_stack([idx_pos, mrs_pos])

    return kin


def main():
    # Get test session IDs
    test_ids = sorted([
        f.name.replace("_sbp.npy", "")
        for f in TEST_DIR.glob("*_sbp.npy")
    ])
    print(f"Phase 2 test sessions: {len(test_ids)}")

    # Get DANDI assets
    print("Fetching DANDI asset list...")
    assets = get_dandi_assets()
    print(f"DANDI assets: {len(assets)}")

    # Build D### mapping
    d_map = build_d_mapping(assets)

    # Save mapping for reference
    with open(LINK_DIR / "d_mapping.json", "w") as f:
        json.dump({k: v["path"] for k, v in d_map.items()}, f, indent=2)
    print(f"Mapping saved to {LINK_DIR / 'd_mapping.json'}")

    # Download and extract kinematics for test sessions
    extracted = 0
    skipped = 0
    failed = 0

    for i, did in enumerate(test_ids):
        kin_path = TEST_DIR / f"{did}_kinematics.npy"
        if kin_path.exists():
            skipped += 1
            continue

        if did not in d_map:
            print(f"  {did}: NOT FOUND in DANDI mapping!")
            failed += 1
            continue

        asset = d_map[did]
        nwb_path = LINK_DIR / f"{did}.nwb"

        print(f"[{i + 1}/{len(test_ids)}] {did} <- {asset['path']}")

        # Download if not cached
        if not nwb_path.exists():
            download_nwb(asset["asset_id"], nwb_path)

        # Extract kinematics
        kin = extract_kinematics_from_nwb(nwb_path)
        if kin is None:
            failed += 1
            continue

        # Check shape matches SBP
        sbp = np.load(TEST_DIR / f"{did}_sbp.npy")
        if kin.shape[0] != sbp.shape[0]:
            print(f"  WARNING: shape mismatch! SBP={sbp.shape[0]}, kin={kin.shape[0]}")
            # Trim or pad to match
            min_len = min(kin.shape[0], sbp.shape[0])
            kin = kin[:min_len]

        # Normalize positions to [0, 1] if needed
        for c in range(2):
            cmin, cmax = kin[:, c].min(), kin[:, c].max()
            if cmax - cmin > 1.5:  # raw values, need normalization
                kin[:, c] = (kin[:, c] - cmin) / (cmax - cmin + 1e-8)

        np.save(kin_path, kin)
        extracted += 1
        print(f"  Saved: {kin_path.name} shape={kin.shape}")

        # Clean up NWB to save disk space
        if nwb_path.exists():
            nwb_path.unlink()

    print(f"\nDone! Extracted: {extracted}, Skipped (exists): {skipped}, Failed: {failed}")
    print(f"Kinematics files in test dir: {len(list(TEST_DIR.glob('*_kinematics.npy')))}")


if __name__ == "__main__":
    main()
