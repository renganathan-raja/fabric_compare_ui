# Fabric embedding-model comparison UI

Streamlit app that compares two embedding models (Gemini Embedding 2, FashionSigLIP) at three levels
(image, attributes, image + attributes). Tabs: **Search**, **Compare** (one level for both models, each
with its own attributes), **Scorecard** (Precision@k / Hit@k / MRR@k, relevant = same composition).

## Runtime data: `deploy/`

The app reads only `deploy/` (about 37 MB): `catalog.csv`, `fashionclip_attributes.csv`, `gemini.npz`,
`siglip.npz` and 400 px `thumbs/`. Rebuild it from the fabric_discovery master CSVs and images with

```
.venv/bin/python utils/prepare_deploy.py
```

`utils/embed_cropped.py --cropped <dir>` re-embeds the images from swatch-only crops (label removed) for both models,
writes copies of the masters to `data/cropped_embedding/` and updates the image vectors and thumbnails in `deploy/`.

The other scripts in `utils/` produced the FashionSigLIP attributes / embeddings
(`generate_attribute.py`, `generate_image_embedding.py`, `combine_fashionclip.py`) and need the repo.

## Run locally

```
./run.sh        # sets FABRIC_IMAGES_DIR for full-size images; uploads / text queries need the repo + keys
```

## Host on Streamlit Community Cloud (catalog browsing only)

Without the `fabric_discovery` repo the app offers only "Pick from catalog" (no uploads or text queries,
so no API key and no torch). Memory use is about 300 MB.

1. Put the project in a GitHub repo (private is fine): `app.py`, `search.py`, `requirements.txt`,
   `deploy/`. `.gitignore` already leaves out the raw `data/`, `.venv/` and `votes.csv`.
2. share.streamlit.io -> New app -> pick the repo, branch, main file `app.py`; Advanced settings -> Python 3.12.
3. `votes.csv` is written to the app's disk, which is wiped on every restart.
