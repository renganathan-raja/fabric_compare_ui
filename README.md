# Fabric embedding-model comparison UI

Streamlit app that compares two embedding models (Gemini Embedding 2, FashionSigLIP) at three levels
(image, attributes, image + attributes). Tabs: **Search**, **Compare** (one level for both models, each
with its own attributes), **Scorecard** (Precision@k / Hit@k / MRR@k, relevant = same composition).

## Runtime data: `deploy/`

The app reads only `deploy/` (about 37 MB): `catalog.csv`, `fashionclip_attributes.csv`, `gemini.npz`,
`siglip.npz` and 400 px `thumbs/`. Rebuild it from the master CSVs in `data/master/` and the images with

```
.venv/bin/python utils/prepare_deploy.py
```

`utils/embed_cropped.py --cropped <dir>` re-embeds the images from swatch-only crops (label removed) for both models,
writes copies of the masters to `data/cropped_embedding/` and updates the image vectors and thumbnails in `deploy/`.

`data/master/{fashionclip,gemini}_master.csv` hold, per row, the cropped image embeddings, the attribute
embeddings (`attribute_embedding`) and the same attributes without `color_effect`
(`attribute_embedding_no_color`, added by `utils/generate_no_color_embedding.py`, which needs the repo and,
for Gemini, an API key). `deploy/*.npz` carries them as `attr` and `attr_no_color`; the sidebar radio
**Attribute embedding** (With color / Without color) picks which one the Attributes and Image + attributes
levels use, and the choice is saved with each vote (`attribute_embedding` column).

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
3. Votes: without the secrets below they go to `votes.csv` on the app's disk, which is wiped on every
   restart. To keep them, store them in a Google Sheet (no Google Cloud needed):
   - Create an empty Google Sheet -> Extensions -> Apps Script -> paste `utils/votes_apps_script.js`.
   - Project Settings (gear) -> Script properties -> add `SECRET` = a long random string.
   - Deploy -> New deployment -> type **Web app**, Execute as **Me**, Who has access **Anyone** -> Deploy,
     authorise when asked, copy the web app URL (ends in `/exec`).
   - Streamlit Cloud -> app -> Settings -> Secrets:
     ```toml
     votes_webhook_url = "https://script.google.com/macros/s/<id>/exec"
     votes_webhook_secret = "<the same string as SECRET>"
     ```
   After editing the script, use Deploy -> Manage deployments -> edit -> New version, or the URL keeps
   serving the old code.
