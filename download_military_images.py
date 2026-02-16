#!/usr/bin/env python3
"""
Download military equipment images from Pexels and Roboflow into military_training folders.
Usage:
  pip install -r requirements_download.txt
  python download_military_images.py --pexels-key KEY --roboflow-key KEY
  python download_military_images.py --source pexels --pexels-key KEY
  python download_military_images.py --source roboflow --roboflow-key KEY
"""

import argparse
import os
import shutil
import time
from pathlib import Path

# Pexels: folder -> search queries (use first query per batch)
PEXELS_QUERIES = {
    "bpla": ["military drone", "UAV", "quadcopter"],
    "aircrafts": ["fighter jet", "military aircraft", "warplane"],
    "birds": ["bird flying", "bird in sky"],
    "tanks": ["military tank", "armored vehicle"],
    "helicopters": ["military helicopter", "attack helicopter"],
    "ships": ["warship", "naval ship", "military vessel"],
    "apc": ["APC military", "armored personnel carrier"],
    "trucks": ["military truck", "army truck"],
    "artillery": ["military artillery", "self-propelled gun", "howitzer"],
    "submarines": ["military submarine", "naval submarine", "attack submarine"],
}

# Roboflow class name -> our folder
ROBOFLOW_CLASS_TO_FOLDER = {
    "drone": "bpla",
    "uav": "bpla",
    "airplane": "aircrafts",
    "jet": "aircrafts",
    "stealth": "aircrafts",
    "birds": "birds",
    "bird": "birds",
    "tank": "tanks",
    "helicopter": "helicopters",
    "mil_helicopter": "helicopters",
    "ship": "ships",
    "vessel": "ships",
    "apc": "apc",
    "armored": "apc",
    "mil_truck": "trucks",
    "truck": "trucks",
    "hummer": "trucks",
    "artillery": "artillery",
    "howitzer": "artillery",
    "submarine": "submarines",
}

# Roboflow datasets: (workspace, project)
# Fallback class names per project when data.yaml is missing
ROBOFLOW_DATASETS = [
    ("military-xmb2h", "military-drone-detection", ["helicopter", "UAV", "airplane", "birds", "drone"]),
    ("military-vehicle-object-detection", "military-vehicles-object-detection",
     ["drone", "tank", "mil_helicopter", "mil_truck", "jet", "hummer", "stealth", "ship", "apc"]),
]


def download_from_pexels(api_key: str, output_dir: Path, per_class: int) -> int:
    """Download images from Pexels API. Returns count of downloaded images."""
    import requests

    total = 0
    headers = {"Authorization": api_key}

    for folder, queries in PEXELS_QUERIES.items():
        folder_path = output_dir / folder
        folder_path.mkdir(parents=True, exist_ok=True)
        existing = len(list(folder_path.glob("*.jpg"))) + len(list(folder_path.glob("*.jpeg"))) + len(list(folder_path.glob("*.png")))
        needed = max(0, per_class - existing)
        if needed <= 0:
            print(f"  [Pexels] {folder}: already has enough ({existing})")
            continue

        collected = 0
        page = 1
        for query in queries:
            if collected >= needed:
                break
            while collected < needed:
                url = "https://api.pexels.com/v1/search"
                params = {"query": query, "per_page": min(80, needed - collected), "page": page}
                try:
                    r = requests.get(url, headers=headers, params=params, timeout=30)
                    if r.status_code == 429:
                        wait_sec = 3600
                        print(f"  [Pexels] Rate limit (429). Waiting {wait_sec}s...")
                        time.sleep(wait_sec)
                        continue
                    r.raise_for_status()
                    data = r.json()
                except Exception as e:
                    print(f"  [Pexels] Error for '{query}': {e}")
                    break

                photos = data.get("photos", [])
                if not photos:
                    break

                for i, photo in enumerate(photos):
                    if collected >= needed:
                        break
                    src = photo.get("src", {})
                    img_url = src.get("original") or src.get("large2x") or src.get("large") or src.get("medium")
                    if not img_url:
                        continue
                    ext = ".jpg"
                    if ".png" in img_url.lower():
                        ext = ".png"
                    out_path = folder_path / f"pexels_{photo.get('id', collected)}_{collected}{ext}"
                    try:
                        img_r = requests.get(img_url, timeout=30)
                        img_r.raise_for_status()
                        with open(out_path, "wb") as f:
                            f.write(img_r.content)
                        collected += 1
                        total += 1
                        if collected % 10 == 0:
                            print(f"  [Pexels] {folder}: {collected}/{needed}")
                    except Exception as e:
                        pass  # skip failed downloads
                    time.sleep(1.0)  # rate limit (Pexels ~200 req/hour)
                page += 1
                if len(photos) < 80:
                    break

        print(f"  [Pexels] {folder}: downloaded {collected} (total in folder: {existing + collected})")

    return total


def download_from_roboflow(api_key: str, output_dir: Path) -> int:
    """Download datasets from Roboflow and organize by class. Returns count of images."""
    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    total = 0
    temp_base = output_dir / "_roboflow_temp"
    temp_base.mkdir(parents=True, exist_ok=True)

    for item in ROBOFLOW_DATASETS:
        workspace_name = item[0]
        project_name = item[1]
        fallback_classes = item[2] if len(item) > 2 else []
        try:
            project = rf.workspace(workspace_name).project(project_name)
            version = project.version(1)
            download_path = temp_base / f"{workspace_name}_{project_name}"
            print(f"  [Roboflow] Downloading {workspace_name}/{project_name}...")
            version.download("yolov8", location=str(download_path))
        except Exception as e:
            print(f"  [Roboflow] Skip {workspace_name}/{project_name}: {e}")
            continue

        # Parse YOLO structure: train/valid/test with images/ and labels/
        class_names = _get_roboflow_class_names(download_path) or fallback_classes
        for split in ["train", "valid", "val", "test"]:
            images_dir = download_path / split / "images"
            labels_dir = download_path / split / "labels"
            if not images_dir.exists():
                images_dir = download_path / split
                labels_dir = download_path / split
            if not images_dir.exists():
                continue

            for img_path in list(images_dir.glob("*")):
                if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                    continue
                label_path = (labels_dir / img_path.stem).with_suffix(".txt")
                if not label_path.exists():
                    label_path = (labels_dir / img_path.name).with_suffix(".txt")

                target_folder = None
                if label_path.exists():
                    # YOLO: class_id x_center y_center w h (normalized)
                    best_class = None
                    best_area = 0.0
                    with open(label_path) as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                try:
                                    cls_id = int(parts[0])
                                    w = float(parts[3])
                                    h = float(parts[4])
                                    area = w * h
                                    if area > best_area:
                                        best_area = area
                                        best_class = cls_id
                                except (ValueError, IndexError):
                                    pass

                    if best_class is not None and best_class < len(class_names):
                        cls_name = class_names[best_class].lower().replace(" ", "_").replace("-", "_")
                        target_folder = ROBOFLOW_CLASS_TO_FOLDER.get(cls_name)
                        if not target_folder:
                            for k, v in ROBOFLOW_CLASS_TO_FOLDER.items():
                                if k in cls_name or cls_name in k:
                                    target_folder = v
                                    break
                else:
                    target_folder = "aircrafts"  # fallback

                if target_folder:
                    dest_dir = output_dir / target_folder
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    dest_path = dest_dir / f"roboflow_{workspace_name}_{project_name}_{img_path.stem}{img_path.suffix}"
                    if not dest_path.exists() or dest_path.stat().st_size == 0:
                        shutil.copy2(img_path, dest_path)
                        total += 1

    if temp_base.exists():
        shutil.rmtree(temp_base, ignore_errors=True)

    return total


def _get_roboflow_class_names(download_path: Path) -> list:
    """Read class names from data.yaml (YOLO format)."""
    for candidate in [download_path / "data.yaml", download_path / "dataset.yaml"]:
        if candidate.exists():
            try:
                import yaml
                with open(candidate) as f:
                    data = yaml.safe_load(f)
                names = data.get("names", {})
                if isinstance(names, dict):
                    n = max(names.keys()) + 1 if names else 0
                    return [names.get(i, f"class_{i}") for i in range(n)]
                if isinstance(names, list):
                    return names
            except Exception:
                pass
    return []


def main():
    parser = argparse.ArgumentParser(description="Download military equipment images for training")
    parser.add_argument("--pexels-key", default=os.environ.get("PEXELS_API_KEY"), help="Pexels API key")
    parser.add_argument("--roboflow-key", default=os.environ.get("ROBOFLOW_API_KEY"), help="Roboflow API key")
    parser.add_argument("--source", choices=["pexels", "roboflow", "both"], default="both")
    parser.add_argument("--per-class", type=int, default=40, help="Images per class from Pexels")
    parser.add_argument("--output-dir", default="military_training", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total = 0

    if args.source in ("pexels", "both") and args.pexels_key:
        print("[Pexels] Downloading...")
        total += download_from_pexels(args.pexels_key, output_dir, args.per_class)
    elif args.source in ("pexels", "both") and not args.pexels_key:
        print("[Pexels] Skip: no API key. Use --pexels-key or set PEXELS_API_KEY")

    if args.source in ("roboflow", "both") and args.roboflow_key:
        print("[Roboflow] Downloading...")
        total += download_from_roboflow(args.roboflow_key, output_dir)
    elif args.source in ("roboflow", "both") and not args.roboflow_key:
        print("[Roboflow] Skip: no API key. Use --roboflow-key or set ROBOFLOW_API_KEY")

    print(f"\nDone. Total images: {total}")
    print(f"Output: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
