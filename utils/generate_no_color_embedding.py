"""Add attribute embeddings WITHOUT `color_effect` to the masters and to deploy/*.npz.

For each model the row's attributes are rebuilt as "key: value; ..." text over every attribute except
`color_effect` (same formatting and filtering as the original `attribute_embedding`) and embedded by
that model's text tower. Written as
  - column `attribute_embedding_no_color` in data/<model>_embedding/<master>.csv (existing columns kept);
  - array `attr_no_color` + mask `has_attr_no_color` in deploy/gemini.npz / deploy/siglip.npz
    (every other array, incl. the original `attr`, is left untouched).
Each distinct text is embedded once. Vectors are unit length.

Needs the fabric_discovery checkout on the branch that has utils/fashionclip_embedding.py and
utils/embedding_client.py (feature/image-texture-agents), and a Google key for --models gemini.

Run from the project folder:
    .venv/bin/python utils/generate_no_color_embedding.py [--models siglip gemini]
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

csv.field_size_limit(sys.maxsize)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import search as S  # noqa: E402

DEPLOY = ROOT / "deploy"
MASTERS = {
    "gemini": ROOT / "data" / "gemini_embedding" / "fabric_master_embeddings_v2.csv",
    "siglip": ROOT / "data" / "fashionclip_embedding" / "fashionclip_master.csv",
}
COLUMN = "attribute_embedding_no_color"
KEYS = [a for a in S.ATTRIBUTES if a != "color_effect"]


def vector_text(vector: np.ndarray) -> str:
    return "[" + ",".join(f"{v:.8g}" for v in vector) + "]"


def parse(text: str) -> np.ndarray:
    return np.array(text.strip().strip("[]").split(","), dtype=np.float32)


def gemini_text(row: dict[str, str]) -> str:
    """Same as embed_attributes_text: skip empty and "unsure" values ("none" is kept)."""
    parts = [f"{k}: {row[k].strip()}" for k in KEYS if row.get(k, "").strip().lower() not in ("", "unsure")]
    return "; ".join(parts)


def embedder(kind: str):
    if kind == "siglip":
        from fabric_discovery.utils.fashionclip_embedding import FashionSigLIP

        model = FashionSigLIP()
        return lambda texts: np.array(model.embed_texts(texts), dtype=np.float32)
    from google.genai import types

    from fabric_discovery.config.settings import settings
    from fabric_discovery.utils.embedding_client import build_client

    client = build_client()

    def embed(texts: list[str]) -> np.ndarray:
        out = []
        for n, text in enumerate(texts, 1):
            result = client.models.embed_content(
                model=settings.embedding_model_name,
                contents=text,
                config=types.EmbedContentConfig(output_dimensionality=settings.embedding_dimensions),
            )
            out.append(result.embeddings[0].values)
            if n % 100 == 0:
                print(f"gemini {n}/{len(texts)}", flush=True)
        return np.array(out, dtype=np.float32)

    return embed


def run(kind: str) -> None:
    path = MASTERS[kind]
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = [f for f in (reader.fieldnames or []) if f != COLUMN]

    if kind == "siglip":
        from fabric_discovery.utils.fashionclip_embedding import attribute_text

        texts = [attribute_text(row, KEYS) for row in rows]
    else:
        texts = [gemini_text(row) for row in rows]
    distinct = list(dict.fromkeys(t for t in texts if t))
    print(f"{kind}: {len(distinct)} distinct texts to embed", flush=True)

    vectors = S.normalise(embedder(kind)(distinct))
    lookup = {t: vector_text(v) for t, v in zip(distinct, vectors, strict=True)}
    for row, text in zip(rows, texts, strict=True):
        row[COLUMN] = lookup.get(text, "")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*fields, COLUMN])
        writer.writeheader()
        writer.writerows(rows)

    # deploy/<kind>.npz: first row of each image that has the vector (as prepare_deploy.py does)
    npz_path = DEPLOY / f"{kind}.npz"
    with np.load(npz_path, allow_pickle=True) as npz:
        data = {k: npz[k] for k in npz.files}
    position = {str(i): n for n, i in enumerate(data["ids"])}
    attr = np.zeros_like(data["attr"], dtype=np.float32)
    has = np.zeros(len(position), dtype=bool)
    for row in rows:
        n = position.get(row["image_id"])
        if n is not None and not has[n] and row[COLUMN]:
            attr[n], has[n] = parse(row[COLUMN]), True
    data["attr_no_color"] = S.normalise(attr).astype(np.float16)
    data["has_attr_no_color"] = has
    np.savez_compressed(npz_path, **data)
    print(f"{kind}: {int(has.sum())}/{len(has)} images have a no-color attribute vector")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--models", nargs="+", choices=list(MASTERS), default=list(MASTERS))
    for kind in parser.parse_args().models:
        run(kind)


if __name__ == "__main__":
    main()
