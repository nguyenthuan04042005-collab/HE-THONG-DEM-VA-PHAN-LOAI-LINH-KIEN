"""Generate simple separated components to verify the algorithms.

These flat synthetic silhouettes are a debugging aid, not evidence of accuracy
on photographs. Pillow drawing is used only for creating input/ground truth.
"""

import argparse
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def make_scene(seed=0):
    rng = np.random.default_rng(seed)
    width, height = 640, 480
    background, foreground = (240, 240, 240), (65, 65, 65)
    image = Image.new("RGB", (width, height), background)
    objects = []
    for index, (cx, cy) in enumerate([(110, 120), (320, 120), (530, 120),
                                     (110, 350), (320, 350), (530, 350)]):
        label = ("bolt", "washer", "nut")[index % 3]
        mask_image = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask_image)
        angle = float(rng.uniform(0, 2 * math.pi))
        if label == "bolt":
            length = int(rng.integers(105, 135))
            points = [(-12, -length / 2), (12, -length / 2), (12, length / 2 - 14),
                      (23, length / 2 - 14), (23, length / 2), (-23, length / 2),
                      (-23, length / 2 - 14), (-12, length / 2 - 14)]
            rotated = [(cx + x * math.cos(angle) - y * math.sin(angle),
                        cy + x * math.sin(angle) + y * math.cos(angle)) for x, y in points]
            draw.polygon(rotated, fill=255)
        else:
            radius = int(rng.integers(35, 47))
            if label == "washer":
                draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=255)
            else:
                points = [(cx + radius * math.cos(angle + k * math.pi / 3),
                           cy + radius * math.sin(angle + k * math.pi / 3)) for k in range(6)]
                draw.polygon(points, fill=255)
            hole_radius = int(radius * float(rng.uniform(0.32, 0.45)))
            draw.ellipse((cx - hole_radius, cy - hole_radius, cx + hole_radius, cy + hole_radius), fill=0)
        image.paste(foreground, (0, 0), mask_image)
        bbox = mask_image.getbbox()
        objects.append({"label": label, "bbox": list(bbox)})
    return image, objects


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/synthetic")
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    entries = []
    for index in range(args.count):
        image, objects = make_scene(args.seed + index)
        filename = f"scene_{index:03d}.png"
        image.save(output / filename)
        entries.append({"image": filename, "objects": objects})
    annotations = {"description": "Synthetic debugging data; not a photographic benchmark", "images": entries}
    (output / "annotations.json").write_text(json.dumps(annotations, indent=2), encoding="utf-8")
    print(f"Saved {args.count} scenes and annotations to {output}")


if __name__ == "__main__":
    main()
