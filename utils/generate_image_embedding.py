"""Embed every catalog image with FashionSigLIP and save the vectors next to the attributes CSV.

Output: data/fashionclip_embedding/fashionclip_image_embeddings.csv with columns
image_id, embedding_model, dimension, image_embedding (unit-length vector as "[v1,v2,...]").

Run from the project folder:  .venv/bin/python utils/generate_image_embedding.py
"""

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import search as S  # noqa: E402

from fabric_discovery.config.settings import settings  # noqa: E402

from fabric_discovery.utils.fashionclip_embedding import FashionSigLIP  # noqa: E402

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "fashionclip_embedding" / "fashionclip_image_embeddings.csv"
S.IMAGES_DIR = settings.resolve_path(settings.fabric_images_dir)  # full-size images, not thumbnails
BATCH = 32


def main() -> None:
    attributes = OUT_PATH.with_name("fashionclip_attributes.csv")
    with attributes.open(encoding="utf-8", newline="") as handle:
        image_ids = list(dict.fromkeys(row["image_id"] for row in csv.DictReader(handle)))

    model = FashionSigLIP()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    with OUT_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_id", "embedding_model", "dimension", "image_embedding"])
        for start in range(0, len(image_ids), BATCH):
            batch = [(i, S.image_path(i)) for i in image_ids[start : start + BATCH]]
            batch = [(i, p) for i, p in batch if p is not None]
            vectors = S.normalise(np.array(model.embed_images([p for _, p in batch]), dtype=np.float32))
            for (image_id, _), vector in zip(batch, vectors, strict=True):
                text = "[" + ",".join(f"{v:.8g}" for v in vector) + "]"
                writer.writerow([image_id, model.model_name, len(vector), text])
            done += len(batch)
            print(f"{done}/{len(image_ids)} images", flush=True)
    print(f"wrote {done} vectors -> {OUT_PATH}")


if __name__ == "__main__":
    main()
