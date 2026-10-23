"""Re-embed the catalog images from their swatch-only crops (label removed), for both models.

Input: a folder of crops named <image_id>.jpg (written by image_cover_extarction/crop_labels.py).
Steps
  1. copies data/gemini_embedding and data/fashionclip_embedding to data/cropped_embedding/ (the
     originals stay untouched) and replaces `image_embedding` in the copies with the crop vectors;
  2. updates the `image` vectors in deploy/gemini.npz and deploy/siglip.npz and the deploy/thumbs/
     thumbnails. Attribute vectors are unchanged (they embed attribute text, not pixels).
Images without a crop keep their old vector and thumbnail.

Run from the project folder:
    .venv/bin/python utils/embed_cropped.py --cropped <dir> [--limit N]
"""

import argparse
import csv
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

csv.field_size_limit(sys.maxsize)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import search as S  # noqa: E402

from fabric_discovery.utils.embedding_client import build_client, embed_image_bytes  # noqa: E402
from fabric_discovery.utils.fashionclip_embedding import FashionSigLIP  # noqa: E402

DEPLOY = ROOT / "deploy"
DATA = ROOT / "data"
OUT = DATA / "cropped_embedding"
MASTERS = {
    "gemini": ("gemini_embedding", "fabric_master_embeddings_v2.csv"),
    "siglip": ("fashionclip_embedding", "fashionclip_master.csv"),
}
THUMB_SIZE = 400
BATCH = 32


def vec_text(vector: np.ndarray) -> str:
    return "[" + ",".join(f"{v:.8g}" for v in vector) + "]"


def embed_gemini(paths: dict[str, Path], workers: int) -> dict[str, np.ndarray]:
    client = build_client()

    def one(item):
        image_id, path = item
        vector = np.array(embed_image_bytes(client, path.read_bytes(), "image/jpeg"), dtype=np.float32)
        return image_id, vector

    done = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for n, (image_id, vector) in enumerate(pool.map(one, paths.items()), 1):
            done[image_id] = vector
            if n % 50 == 0:
                print(f"gemini {n}/{len(paths)}", flush=True)
    return done


def embed_siglip(paths: dict[str, Path]) -> dict[str, np.ndarray]:
    model = FashionSigLIP()
    ids, done = list(paths), {}
    for start in range(0, len(ids), BATCH):
        batch = ids[start : start + BATCH]
        vectors = np.array(model.embed_images([paths[i] for i in batch]), dtype=np.float32)
        done.update(zip(batch, vectors, strict=True))
        print(f"siglip {len(done)}/{len(ids)}", flush=True)
    return done


def rewrite_master(kind: str, vectors: dict[str, np.ndarray]) -> None:
    folder, name = MASTERS[kind]
    (OUT / folder).mkdir(parents=True, exist_ok=True)
    for extra in (DATA / folder).iterdir():
        if extra.name != name:
            shutil.copy2(extra, OUT / folder / extra.name)
    with (DATA / folder / name).open(encoding="utf-8-sig", newline="") as src, (OUT / folder / name).open(
        "w", encoding="utf-8", newline=""
    ) as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            vector = vectors.get(row["image_id"])
            if vector is not None:
                row["image_embedding"] = vec_text(vector)
            writer.writerow(row)


def update_deploy(kind: str, vectors: dict[str, np.ndarray]) -> None:
    path = DEPLOY / f"{kind}.npz"
    with np.load(path, allow_pickle=True) as npz:
        data = {k: npz[k] for k in npz.files}
    position = {i: n for n, i in enumerate(data["ids"])}
    image = data["image"].astype(np.float32)
    for image_id, vector in vectors.items():
        n = position[image_id]
        image[n] = S.normalise(vector[None])[0]
        data["has_image"][n] = True
    data["image"] = image.astype(np.float16)
    np.savez_compressed(path, **data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cropped", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-deploy", action="store_true", help="only write data/cropped_embedding")
    args = parser.parse_args()

    with (DEPLOY / "catalog.csv").open(encoding="utf-8", newline="") as handle:
        catalog_ids = [r["image_id"] for r in csv.DictReader(handle)]
    paths = {i: args.cropped / f"{i}.jpg" for i in catalog_ids if (args.cropped / f"{i}.jpg").exists()}
    if args.limit:
        paths = dict(list(paths.items())[: args.limit])
    print(f"{len(paths)} of {len(catalog_ids)} catalog images have a crop")

    results = {"gemini": embed_gemini(paths, args.workers), "siglip": embed_siglip(paths)}
    for kind, vectors in results.items():
        rewrite_master(kind, vectors)
        if not args.no_deploy:
            update_deploy(kind, vectors)
    if not args.no_deploy:
        for image_id, path in paths.items():
            with Image.open(path) as img:
                img = img.convert("RGB")
                img.thumbnail((THUMB_SIZE, THUMB_SIZE))
                img.save(DEPLOY / "thumbs" / f"{image_id}.jpg", quality=82, optimize=True)
    print(f"done -> {OUT}" + ("" if args.no_deploy else " and deploy/"))


if __name__ == "__main__":
    main()
