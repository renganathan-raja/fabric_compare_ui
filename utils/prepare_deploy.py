"""Build the self-contained `deploy/` bundle the app (and Streamlit Cloud) runs from.

Reads the two master CSVs in data/master/ (cropped image vectors, attribute vectors with and without
color_effect; built from the fabric_discovery masters by utils/embed_cropped.py and
utils/generate_no_color_embedding.py), and writes
  deploy/catalog.csv                one row per image: image_id, rd_number, material_id, composition, 18 attributes
  deploy/fashionclip_attributes.csv one row per image: image_id + the FashionSigLIP-predicted attributes
  deploy/gemini.npz, siglip.npz     ids, image / attr / attr_no_color vectors (float16, unit length) and
                                    presence masks (has_image, has_attr, has_attr_no_color)
                                    (siglip attr = text embedding of FashionSigLIP's predicted attributes)
  deploy/thumbs/<image_id>.jpg      400 px thumbnails of the catalog images

Run from the project folder:  .venv/bin/python utils/prepare_deploy.py
"""

import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

csv.field_size_limit(sys.maxsize)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import search as S  # noqa: E402

from fabric_discovery.config.settings import settings  # noqa: E402

DEPLOY = ROOT / "deploy"
MASTER_DIR = ROOT / "data" / "master"
THUMB_SIZE = 400
BLANK = ("", "null", "nan")


def read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse(text: str) -> np.ndarray:
    return np.array(text.strip().strip("[]").split(","), dtype=np.float32)


def main() -> None:
    DEPLOY.mkdir(exist_ok=True)
    masters = {
        "gemini": read(MASTER_DIR / "gemini_master.csv"),
        # FashionSigLIP: its attribute embeddings embed FashionSigLIP's own predicted attributes
        "siglip": read(MASTER_DIR / "fashionclip_master.csv"),
    }

    # catalog metadata: Gemini master, one row per image (first non-blank composition, all rd_numbers)
    rows = masters["gemini"]
    image_ids = list(dict.fromkeys(r["image_id"] for r in rows))
    catalog = []
    for image_id in image_ids:
        group = [r for r in rows if r["image_id"] == image_id]
        first = group[0]
        composition = next((r["composition"] for r in group if r["composition"].strip().lower() not in BLANK), "")
        rds = list(dict.fromkeys(r["rd_number"] for r in group if r["rd_number"]))
        catalog.append(
            {"image_id": image_id, "rd_number": "|".join(rds), "material_id": first["material_id"],
             "composition": composition, **{a: first.get(a, "") for a in S.ATTRIBUTES}}
        )
    with (DEPLOY / "catalog.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(catalog[0]))
        writer.writeheader()
        writer.writerows(catalog)

    # FashionSigLIP-predicted attributes (one row per image)
    seen = {}
    for r in masters["siglip"]:
        seen.setdefault(r["image_id"], {"image_id": r["image_id"], **{a: r.get(a, "") for a in S.ATTRIBUTES}})
    with (DEPLOY / "fashionclip_attributes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", *S.ATTRIBUTES])
        writer.writeheader()
        writer.writerows(seen.values())

    # vectors per model: first row of each image that has the vector
    for kind, model_rows in masters.items():
        dim = len(parse(next(r["image_embedding"] for r in model_rows if r["image_embedding"])))
        image = np.zeros((len(image_ids), dim), dtype=np.float32)
        attr = np.zeros((len(image_ids), dim), dtype=np.float32)
        has_image = np.zeros(len(image_ids), dtype=bool)
        has_attr = np.zeros(len(image_ids), dtype=bool)
        attr_no_color = np.zeros((len(image_ids), dim), dtype=np.float32)
        has_attr_no_color = np.zeros(len(image_ids), dtype=bool)
        position = {i: n for n, i in enumerate(image_ids)}
        for r in model_rows:
            n = position.get(r["image_id"])
            if n is None:
                continue
            if not has_image[n] and r["image_embedding"]:
                image[n], has_image[n] = parse(r["image_embedding"]), True
            if not has_attr_no_color[n] and r.get("attribute_embedding_no_color"):
                attr_no_color[n], has_attr_no_color[n] = parse(r["attribute_embedding_no_color"]), True
            if not has_attr[n] and r["attribute_embedding"]:
                attr[n], has_attr[n] = parse(r["attribute_embedding"]), True
        np.savez_compressed(
            DEPLOY / f"{kind}.npz", ids=np.array(image_ids), has_image=has_image, has_attr=has_attr,
            image=S.normalise(image).astype(np.float16), attr=S.normalise(attr).astype(np.float16),
            has_attr_no_color=has_attr_no_color, attr_no_color=S.normalise(attr_no_color).astype(np.float16),
        )
        print(kind, dim, "images with image vector", int(has_image.sum()), "attr vector", int(has_attr.sum()), "attr_no_color", int(has_attr_no_color.sum()))

    # thumbnails
    thumbs = DEPLOY / "thumbs"
    thumbs.mkdir(exist_ok=True)
    made = 0
    for image_id in image_ids:
        if (thumbs / f"{image_id}.jpg").exists():
            made += 1
            continue
        source = next((p for ext in (".jpg", ".jpeg", ".png", ".webp") if (p := settings.resolve_path(settings.fabric_images_dir) / f"{image_id}{ext}").exists()), None)
        if source is None:
            continue
        with Image.open(source) as img:
            img = img.convert("RGB")
            img.thumbnail((THUMB_SIZE, THUMB_SIZE))
            img.save(thumbs / f"{image_id}.jpg", quality=82, optimize=True)
        made += 1
    print(f"{len(catalog)} images in catalog.csv, {made} thumbnails -> {DEPLOY}")


if __name__ == "__main__":
    main()
