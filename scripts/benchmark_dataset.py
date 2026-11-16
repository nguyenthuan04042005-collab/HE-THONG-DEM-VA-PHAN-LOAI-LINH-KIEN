"""Run a frozen configuration and compare per-class counts with manual labels.

Count labels are independent human observations of the original photographs.
This report measures count correctness, not bbox IoU or per-object accuracy.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from time import perf_counter

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source_code"))
from config import load_config
from pipeline import analyze_image, save_result

CLASSES = ("bolt", "washer", "nut")


def natural_key(path):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(path))]


def contact_sheets(entries, output, overlay=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 19)
    tile_w, tile_h = 360, 520
    for start in range(0, len(entries), 9):
        batch = entries[start:start + 9]
        canvas = Image.new("RGB", (tile_w * 3, tile_h * math.ceil(len(batch) / 3)), "#d7dce2")
        draw = ImageDraw.Draw(canvas)
        for index, entry in enumerate(batch):
            x, y = index % 3 * tile_w, index // 3 * tile_h
            path = entry["overlay"] if overlay else entry["path"]
            with Image.open(path) as source:
                image = source.convert("RGB")
                image.thumbnail((tile_w - 12, tile_h - 55))
            canvas.paste(image, (x + (tile_w - image.width) // 2, y + 49))
            title = entry["name"]
            draw.text((x + 8, y + 3), title, fill="black", font=font)
            if overlay:
                counts = entry["counts"]
                text = f"B:{counts['bolt']} W:{counts['washer']} N:{counts['nut']} ?:{counts['unknown']}"
                draw.text((x + 8, y + 24), text, fill="black", font=font)
        canvas.save(output / f"{'predictions' if overlay else 'originals'}_{start // 9 + 1:02d}.jpg", quality=93)


def prepare(input_dir, output, config_path):
    input_dir, output = Path(input_dir).resolve(), Path(output).resolve()
    files = sorted((p for p in input_dir.iterdir() if p.suffix.lower() in
                    (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")), key=natural_key)
    if not files:
        raise ValueError("No images found")
    config = load_config(config_path)
    source_files = sorted((ROOT / "source_code").glob("*.py"))
    source_manifest = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}
    source_hash = hashlib.sha256(json.dumps(source_manifest, sort_keys=True).encode("utf-8")).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    entries = []
    for path in files:
        with Image.open(path) as image:
            size = list(image.size)
        entries.append({"name": path.name, "path": str(path), "size": size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    contact_sheets(entries, output / "review_originals")
    (output / "inventory.json").write_text(json.dumps(entries, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps(source_manifest, indent=2), encoding="utf-8")
    predictions = []
    start = perf_counter()
    for index, entry in enumerate(entries):
        target = output / "images" / Path(entry["name"]).stem
        report_file = target / "summary.json"
        if report_file.exists():
            old = json.loads(report_file.read_text(encoding="utf-8"))
        else:
            old = None
        if (old and old.get("parameters") == config.to_dict() and old.get("input_sha256") == entry["sha256"]
                and old.get("source_sha256") == source_hash):
            report = old
        else:
            report, stages = analyze_image(entry["path"], config)
            report["input_sha256"] = entry["sha256"]
            report["source_sha256"] = source_hash
            save_result(report, stages, target, overview=False)
        predictions.append({**entry, "counts": report["counts"], "total": report["total"],
                            "seconds": report["processing_seconds"], "warnings": report["warnings"],
                            "overlay": str(target / "overlay.png")})
        print(f"[{index + 1}/{len(entries)}] {entry['name']} {report['counts']}", flush=True)
    contact_sheets(predictions, output / "review_predictions", overlay=True)
    manifest = {"input": str(input_dir), "parameters": config.to_dict(), "images": predictions,
                "source_sha256": source_hash, "elapsed_seconds": round(perf_counter() - start, 2)}
    (output / "predictions.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with (output / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image", "bolt", "washer", "nut", "unknown", "total", "seconds"])
        writer.writeheader()
        for entry in predictions:
            writer.writerow({"image": entry["name"], **entry["counts"], "total": entry["total"], "seconds": entry["seconds"]})
    return manifest


def compare(output, annotation_path):
    output = Path(output)
    manifest = json.loads((output / "predictions.json").read_text(encoding="utf-8"))
    annotations = json.loads(Path(annotation_path).read_text(encoding="utf-8-sig"))
    labels = {entry["image"]: entry for entry in annotations["images"]}
    expected = {entry["name"] for entry in manifest["images"]}
    if set(labels) != expected or len(labels) != len(annotations["images"]):
        raise ValueError("Manual annotations must cover every image exactly once")
    records = []
    totals_true, totals_predicted = Counter(), Counter()
    for entry in manifest["images"]:
        truth = labels[entry["name"]]["counts"]
        if set(truth) != set(CLASSES) or any(not isinstance(v, int) or v < 0 for v in truth.values()):
            raise ValueError("Each annotation needs three nonnegative integer counts")
        predicted = entry["counts"]
        exact = all(truth[name] == predicted[name] for name in CLASSES) and predicted["unknown"] == 0
        record = {"image": entry["name"], "exact_class_counts": exact,
                  "truth_total": sum(truth.values()), "predicted_total": entry["total"],
                  "unknown": predicted["unknown"], "notes": labels[entry["name"]].get("notes", "")}
        for name in CLASSES:
            record.update({f"truth_{name}": truth[name], f"predicted_{name}": predicted[name],
                           f"absolute_error_{name}": abs(truth[name] - predicted[name])})
        record["absolute_class_count_error"] = sum(record[f"absolute_error_{c}"] for c in CLASSES) + predicted["unknown"]
        records.append(record)
        totals_true.update(truth)
        totals_predicted.update(predicted)
    n = len(records)
    exact_n = sum(r["exact_class_counts"] for r in records)
    total_exact = sum(r["truth_total"] == r["predicted_total"] for r in records)
    metrics = {"evaluation_type": "Manual per-class counts; not per-object bbox evaluation",
               "annotation_file": str(Path(annotation_path).resolve()), "images": n,
               "parameters": manifest["parameters"], "truth_counts": dict(totals_true),
               "predicted_counts": dict(totals_predicted),
               "exact_class_counts_images": exact_n, "exact_class_counts_percent": 100 * exact_n / n,
               "exact_total_count_images": total_exact, "exact_total_count_percent": 100 * total_exact / n,
               "per_class_exact_image_percent": {c: 100 * sum(r[f"truth_{c}"] == r[f"predicted_{c}"] for r in records) / n for c in CLASSES},
               "count_mae_per_class": {c: sum(r[f"absolute_error_{c}"] for r in records) / n for c in CLASSES},
               "total_count_mae": sum(abs(r["truth_total"] - r["predicted_total"]) for r in records) / n,
               "failed_images": [r["image"] for r in records if not r["exact_class_counts"]],
               "mean_processing_seconds": sum(e["seconds"] for e in manifest["images"]) / n}
    # Also record the first 43 numbered files, addressing the user's stated size.
    first43_names = {f"anh{i}.jpg" for i in range(1, 44)}
    first43 = [r for r in records if r["image"] in first43_names]
    if first43_names <= expected:
        metrics["anh1_to_anh43"] = {"images": 43, "exact_class_counts_images": sum(r["exact_class_counts"] for r in first43),
                                      "exact_class_counts_percent": 100 * sum(r["exact_class_counts"] for r in first43) / 43}
    groups = {"separated": [], "touching_or_overlapping": []}
    for record in records:
        condition = labels[record["image"]].get("conditions", [])
        group = "touching_or_overlapping" if "touching_or_overlapping" in condition else "separated"
        groups[group].append(record)
    metrics["scene_groups"] = {
        group: {"images": len(items), "exact_class_counts_images": sum(r["exact_class_counts"] for r in items),
                "exact_class_counts_percent": 100 * sum(r["exact_class_counts"] for r in items) / len(items)}
        for group, items in groups.items() if items}
    (output / "count_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "count_comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    group_values = [metrics["exact_class_counts_percent"]] + [g["exact_class_counts_percent"] for g in metrics["scene_groups"].values()]
    group_names = [f"All images (n={n})"] + [f"{name}\n(n={group['images']})" for name, group in metrics["scene_groups"].items()]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(group_names, group_values, color=["#3267b5", "#21866d", "#c65353"][:len(group_names)])
    ax.set_ylim(0, 110)
    ax.set_ylabel("Images with all class counts correct (%)")
    ax.set_title("Frozen baseline on dataXuLiAnh — manual count labels")
    for bar, value in zip(bars, group_values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.2f}%", ha="center")
    fig.tight_layout()
    fig.savefig(output / "accuracy_by_scene.png", dpi=160)
    plt.close(fig)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.json"))
    parser.add_argument("--annotations")
    args = parser.parse_args()
    if args.input:
        prepare(args.input, args.output, args.config)
    if args.annotations:
        compare(args.output, args.annotations)
    if not args.input and not args.annotations:
        parser.error("Provide --input or --annotations")


if __name__ == "__main__":
    main()
