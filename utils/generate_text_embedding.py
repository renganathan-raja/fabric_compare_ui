"""Add FashionSigLIP text embeddings to data/fashionclip_embedding/fashionclip_master.csv.

Two columns are appended (existing columns and row order are untouched):
  - attribute_embedding: the row's FashionSigLIP-predicted attributes as "key: value; ..." text
    (empty / "none" / "unsure" values left out), embedded by the text tower. Blank if no attribute
    is informative.
  - composition_embedding: the composition string embedded by the text tower (blank stays blank).
Vectors are unit length, 768-d, formatted like image_embedding. Each distinct text is embedded once.

Run from the project folder:  .venv/bin/python utils/generate_text_embedding.py
"""

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import search as S  # noqa: E402

from fabric_discovery.utils.fashionclip_embedding import FashionSigLIP, attribute_text  # noqa: E402

csv.field_size_limit(sys.maxsize)
MASTER = Path(__file__).resolve().parent.parent / "data" / "fashionclip_embedding" / "fashionclip_master.csv"
NEW_COLUMNS = ["composition_embedding", "attribute_embedding"]


def vector_text(vector: np.ndarray) -> str:
    return "[" + ",".join(f"{v:.8g}" for v in vector) + "]"


def main() -> None:
    with MASTER.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = [f for f in (reader.fieldnames or []) if f not in NEW_COLUMNS]

    attr_texts = [attribute_text(row, S.ATTRIBUTES) for row in rows]
    comp_texts = [row["composition"].strip() for row in rows]
    distinct = list(dict.fromkeys(t for t in attr_texts + comp_texts if t))
    print(f"{len(distinct)} distinct texts to embed", flush=True)

    model = FashionSigLIP()
    vectors = S.normalise(np.array(model.embed_texts(distinct), dtype=np.float32))
    lookup = {text: vector_text(vector) for text, vector in zip(distinct, vectors, strict=True)}

    for row, attr_text, comp_text in zip(rows, attr_texts, comp_texts, strict=True):
        row["attribute_embedding"] = lookup.get(attr_text, "")
        row["composition_embedding"] = lookup.get(comp_text, "")
    with MASTER.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*fields, *NEW_COLUMNS])
        writer.writeheader()
        writer.writerows(rows)
    for column in NEW_COLUMNS:
        print(f"{column}: {sum(bool(r[column]) for r in rows)}/{len(rows)} rows filled")


if __name__ == "__main__":
    main()
