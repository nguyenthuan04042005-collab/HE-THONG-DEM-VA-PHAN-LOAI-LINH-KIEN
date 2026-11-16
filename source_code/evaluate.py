"""Evaluate detection, classification, and counts using hand-labeled boxes.

Ground truth paths are relative to the annotation JSON. Matching is greedy by
descending IoU, independent of predicted class, so misclassifications stay visible.
"""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import numpy as np
from classifier import CLASS_NAMES
from config import load_config
from pipeline import analyze_image, save_result

TRUE_CLASSES = CLASS_NAMES[:3]


def box_iou(a, b):
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1e-8)


def match_objects(truth, predicted, threshold=0.5):
    candidates = sorted(((box_iou(gt["bbox"], pred["bbox"]), i, j)
                         for i, gt in enumerate(truth) for j, pred in enumerate(predicted)),
                        reverse=True)
    used_truth, used_prediction, matches = set(), set(), []
    for iou, i, j in candidates:
        if iou >= threshold and i not in used_truth and j not in used_prediction:
            matches.append((i, j, iou))
            used_truth.add(i)
            used_prediction.add(j)
    return (matches, sorted(set(range(len(truth))) - used_truth),
            sorted(set(range(len(predicted))) - used_prediction))


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def _validate_dataset(dataset):
    if not isinstance(dataset, dict) or not isinstance(dataset.get("images"), list) or not dataset["images"]:
        raise ValueError("Annotations require a non-empty 'images' list")
    seen = set()
    for entry in dataset["images"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("image"), str) or not isinstance(entry.get("objects"), list):
            raise ValueError("Each image needs 'image' and 'objects'")
        if entry["image"] in seen:
            raise ValueError("Duplicate image in annotation file")
        seen.add(entry["image"])
        for obj in entry["objects"]:
            box = obj.get("bbox", [])
            if (obj.get("label") not in TRUE_CLASSES or len(box) != 4
                    or not all(isinstance(v, (int, float)) and np.isfinite(v) for v in box)
                    or not (0 <= box[0] < box[2] and 0 <= box[1] < box[3])):
                raise ValueError("Each object needs a valid class and [x0,y0,x1,y1] box")


def evaluate(annotation_path, output_dir, config, iou_threshold=0.5):
    if not 0 < iou_threshold <= 1:
        raise ValueError("IoU threshold must be in (0, 1]")
    annotation_path, output_dir = Path(annotation_path), Path(output_dir)
    dataset = json.loads(annotation_path.read_text(encoding="utf-8-sig"))
    _validate_dataset(dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(CLASS_NAMES) + ["spurious"]
    columns = list(CLASS_NAMES) + ["missed"]
    matrix = {row: {col: 0 for col in columns} for row in rows}
    count_errors = {name: [] for name in CLASS_NAMES}
    total_truth = total_predicted = total_matched = correct_matched = exact_counts = 0
    records = []
    class_stats = {name: {"tp": 0, "fp": 0, "fn": 0, "support": 0} for name in TRUE_CLASSES}
    for image_index, entry in enumerate(dataset["images"]):
        image_path = (annotation_path.parent / entry["image"]).resolve()
        report, stages = analyze_image(image_path, config)
        width, height = report["image_size"]
        truth, predicted = entry["objects"], report["objects"]
        if any(obj["bbox"][2] > width or obj["bbox"][3] > height for obj in truth):
            raise ValueError(f"Ground-truth box lies outside {image_path.name}")
        save_result(report, stages, output_dir / "images" / f"{image_index:03d}_{image_path.stem}", overview=False)
        matches, missed, spurious = match_objects(truth, predicted, iou_threshold)
        total_truth += len(truth)
        total_predicted += len(predicted)
        total_matched += len(matches)
        truth_counts = Counter(obj["label"] for obj in truth)
        exact_counts += int(all(truth_counts[name] == report["counts"][name] for name in CLASS_NAMES))
        record = {"image": entry["image"], "truth_total": len(truth), "predicted_total": len(predicted)}
        for name in CLASS_NAMES:
            error = abs(truth_counts[name] - report["counts"][name])
            count_errors[name].append(error)
            record.update({f"truth_{name}": truth_counts[name], f"predicted_{name}": report["counts"][name],
                           f"absolute_error_{name}": error})
        records.append(record)
        for name in TRUE_CLASSES:
            class_stats[name]["support"] += truth_counts[name]
        for i, j, _ in matches:
            actual, predicted_label = truth[i]["label"], predicted[j]["label"]
            matrix[actual][predicted_label] += 1
            if actual == predicted_label:
                correct_matched += 1
                class_stats[actual]["tp"] += 1
            else:
                class_stats[actual]["fn"] += 1
                if predicted_label in class_stats:
                    class_stats[predicted_label]["fp"] += 1
        for i in missed:
            actual = truth[i]["label"]
            matrix[actual]["missed"] += 1
            class_stats[actual]["fn"] += 1
        for j in spurious:
            predicted_label = predicted[j]["label"]
            matrix["spurious"][predicted_label] += 1
            if predicted_label in class_stats:
                class_stats[predicted_label]["fp"] += 1
    for stats in class_stats.values():
        precision = _ratio(stats["tp"], stats["tp"] + stats["fp"])
        recall = _ratio(stats["tp"], stats["tp"] + stats["fn"])
        stats.update(precision=precision, recall=recall, f1=_ratio(2 * precision * recall, precision + recall))
    metrics = {"annotation_file": str(annotation_path.resolve()), "images": len(records),
               "parameters": config.to_dict(), "iou_threshold": iou_threshold,
               "truth_objects": total_truth, "predicted_objects": total_predicted,
               "detection_precision": _ratio(total_matched, total_predicted),
               "detection_recall": _ratio(total_matched, total_truth),
               "classification_accuracy_on_matched": _ratio(correct_matched, total_matched),
               "exact_count_image_accuracy": exact_counts / len(records),
               "count_mae_per_class": {name: float(np.mean(values)) for name, values in count_errors.items()},
               "class_metrics": class_stats, "confusion_matrix": matrix,
               "confusion_rows": rows, "confusion_columns": columns}
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "counts.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with (output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["truth / prediction", *columns])
        for row in rows:
            writer.writerow([row, *(matrix[row][col] for col in columns)])
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output", default="results/evaluation")
    parser.add_argument("--config")
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()
    metrics = evaluate(args.annotations, args.output, load_config(args.config), args.iou)
    print(json.dumps({key: metrics[key] for key in ("images", "truth_objects", "predicted_objects",
                     "detection_precision", "detection_recall", "classification_accuracy_on_matched",
                     "count_mae_per_class")}, indent=2))


if __name__ == "__main__":
    main()
