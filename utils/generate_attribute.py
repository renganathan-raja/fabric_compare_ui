"""Predict the fabric attributes of every catalog image with FashionSigLIP (zero-shot).

FashionSigLIP only scores image/text similarity, so each attribute is a classification: the image
is compared with one prompt per candidate value ("a fabric with pattern: stripe", ...) and the
closest one wins. Candidate values are the single values seen in the Gemini master CSV. When the
winner's softmax confidence is below --min-confidence the cell is left empty (null).

Output: data/fashionclip_embedding/fashionclip_attributes.csv, one row per row of the Gemini master
CSV with the same columns minus the vectors, `dimension` and `embedding_model`; the 18 attribute
columns are the FashionSigLIP predictions, `extraction_model` is the FashionSigLIP model name.

Run from the project folder:  .venv/bin/python utils/generate_attribute.py [--min-confidence 0.4]
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import search as S  # noqa: E402

from fabric_discovery.config.settings import settings  # noqa: E402
from fabric_discovery.utils.fashionclip_embedding import FashionSigLIP  # noqa: E402

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "fashionclip_embedding" / "fashionclip_attributes.csv"
S.IMAGES_DIR = settings.resolve_path(settings.fabric_images_dir)  # full-size images, not thumbnails
BATCH = 32


def read_master() -> tuple[list[str], list[dict[str, str]]]:
    path = settings.resolve_path(settings.output_dir) / settings.fabric_master_output
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--min-confidence", type=float, default=S.MIN_CONFIDENCE)
    parser.add_argument("--limit", type=int, default=None, help="only the first N images (trial run)")
    args = parser.parse_args()

    fields, rows = read_master()
    image_ids = list(dict.fromkeys(row["image_id"] for row in rows))[: args.limit]
    wanted = set(image_ids)
    rows = [row for row in rows if row["image_id"] in wanted]
    vocab = S.attribute_vocabulary(rows)

    model = FashionSigLIP()
    text_vectors = S.embed_vocabulary(vocab, model.embed_texts)

    predictions: dict[str, dict[str, str]] = {}
    for start in range(0, len(image_ids), BATCH):
        batch = image_ids[start : start + BATCH]
        paths = [S.image_path(i) for i in batch]
        usable = [(i, p) for i, p in zip(batch, paths, strict=True) if p is not None]
        if usable:
            vectors = S.normalise(np.array(model.embed_images([p for _, p in usable]), dtype=np.float32))
            for (image_id, _), vector in zip(usable, vectors, strict=True):
                predictions[image_id] = S.predict_attributes(vector, vocab, text_vectors, args.min_confidence)
        print(f"{min(start + BATCH, len(image_ids))}/{len(image_ids)} images", flush=True)

    drop = {"dimension", "embedding_model"} | {f for f in fields if f.endswith("_embedding")}
    columns = [f for f in fields if f not in drop]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            out = {c: row.get(c, "") for c in columns}
            out["extraction_model"] = model.model_name
            out.update(predictions.get(row["image_id"], {}))
            writer.writerow(out)
    empty = sum(1 for p in predictions.values() for v in p.values() if not v)
    total = sum(len(p) for p in predictions.values())
    print(f"wrote {len(rows)} rows for {len(predictions)} images -> {OUT_PATH}")
    print(f"{empty}/{total} attribute cells left empty (confidence < {args.min_confidence})")


if __name__ == "__main__":
    main()
