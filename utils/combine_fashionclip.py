"""Join the FashionSigLIP attributes CSV and image-embeddings CSV into one file.

Output: data/fashionclip_embedding/fashionclip_master.csv, one row per row of the attributes CSV
(all its columns) plus `dimension` and `image_embedding` looked up by image_id.

Run from the project folder:  .venv/bin/python utils/combine_fashionclip.py
"""

import csv
import sys
from pathlib import Path

csv.field_size_limit(sys.maxsize)
FOLDER = Path(__file__).resolve().parent.parent / "data" / "fashionclip_embedding"


def main() -> None:
    with (FOLDER / "fashionclip_image_embeddings.csv").open(encoding="utf-8", newline="") as handle:
        vectors = {r["image_id"]: r for r in csv.DictReader(handle)}
    with (FOLDER / "fashionclip_attributes.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    missing = sorted({r["image_id"] for r in rows} - set(vectors))
    if missing:
        raise SystemExit(f"{len(missing)} images have no embedding, e.g. {missing[:3]}")
    out_path = FOLDER / "fashionclip_master.csv"
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*fields, "dimension", "image_embedding"])
        writer.writeheader()
        for row in rows:
            vector = vectors[row["image_id"]]
            writer.writerow({**row, "dimension": vector["dimension"], "image_embedding": vector["image_embedding"]})
    print(f"wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
