"""Streamlit UI: search fabrics by image/text and compare embedding models at three levels
(image, attributes, image + attributes).

Run:  ./run.sh   (see README.md)
"""

import csv
import dataclasses
import hashlib
import random
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import search as S

APP_DIR = Path(__file__).resolve().parent
VOTES_CSV = APP_DIR / "votes.csv"

st.set_page_config(page_title="Fabric Discovery – Search & Compare", layout="wide")


# ---- cached resources ----------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading embeddings…")
def _corpus(label: str, weight: float) -> S.Corpus:
    return S.load_corpus(S.MODELS[label], weight)


def corpus_for(label: str) -> S.Corpus:
    fused = S.MODELS[label].level == S.LEVEL_FUSED
    return _corpus(label, st.session_state.get("fuse_w", 0.5) if fused else 0.5)


@st.cache_resource(show_spinner="Loading FashionSigLIP (first time downloads the weights)…")
def siglip_model():
    from fabric_discovery.utils.fashionclip_embedding import FashionSigLIP

    return FashionSigLIP()


@st.cache_data(show_spinner=False)
def thumbnail(image_id: str) -> Image.Image | None:
    path = S.image_path(image_id)
    if path is None:
        return None
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((320, 320))
        return image.copy()


FC_ATTRIBUTES_CSV = S.DEPLOY / "fashionclip_attributes.csv"


@st.cache_data(show_spinner=False)
def fashionclip_attributes() -> tuple[dict[str, dict[str, str]], dict[str, list[str]]]:
    """FashionSigLIP-predicted attributes and the rd_numbers per image_id."""
    with FC_ATTRIBUTES_CSV.open(encoding="utf-8", newline="") as handle:
        attrs = {row["image_id"]: {a: row.get(a, "") for a in S.ATTRIBUTES} for row in csv.DictReader(handle)}
    rds = {i: [r for r in meta.get("rd_number", "").split("|") if r] for i, meta in S.load_catalog().items()}
    return attrs, rds


@st.cache_data(show_spinner=False)
def catalog_hashes() -> dict[str, str]:
    """sha1 of every catalog image file -> image_id, to recognise an upload of a catalog image."""
    out = {}
    for image_id in corpus_for(IMAGE_LABEL).ids:
        path = S.image_path(image_id)
        if path is not None:
            out[hashlib.sha1(path.read_bytes()).hexdigest()] = image_id
    return out


def catalog_match(name: str, data: bytes) -> str | None:
    """image_id of the catalog image this upload is (by file name, else identical bytes)."""
    ids = corpus_for(IMAGE_LABEL).meta
    if Path(name).stem in ids:
        return Path(name).stem
    return catalog_hashes().get(hashlib.sha1(data).hexdigest())


IMAGE_LABEL = next(iter(S.MODELS))  # first model at the image level: lists every catalog image


def identity(image_id: str) -> dict[str, str]:
    """image_id, rd_number(s), material_id and composition of a catalog image as key -> value."""
    meta = S.load_catalog().get(image_id, {})
    rds = [r for r in meta.get("rd_number", "").split("|") if r]
    parts = [c.strip() for c in meta.get("composition", "").split(",") if c.strip()]
    return {
        "image_id": image_id,
        "rd_number": ", ".join(rds) or "—",
        "material_id": meta.get("material_id") or "—",
        "composition": ", ".join(parts) or "—",
    }


def identity_markdown(image_id: str) -> str:
    """The identity as bold-key lines (compact, for result cards)."""
    return "  \n".join(f"**{key}:** {value}" for key, value in identity(image_id).items())

# ---- sidebar: query ------------------------------------------------------------------------

st.sidebar.title("Query")
sources = ["Upload image", "Pick from catalog", "Text"] if S.LIVE else ["Pick from catalog"]
source = st.sidebar.radio("Query source", sources)
upload = None
catalog_id = None
text_query = ""
if source == "Upload image":
    upload = st.sidebar.file_uploader("Fabric image", type=["jpg", "jpeg", "png", "webp"])
elif source == "Pick from catalog":
    voted_ids = {}
    for v in S.load_votes(VOTES_CSV):
        voted_ids.setdefault(v["query_id"], set()).add(v["vote"])
    catalog_id = st.sidebar.selectbox(
        "Image id", corpus_for(IMAGE_LABEL).ids,
        format_func=lambda i: f"{i} (voted)" if f"id:{i}" in voted_ids else i,
    )
    if f"id:{catalog_id}" in voted_ids:
        st.sidebar.caption("✓ voted – see the Human evaluation tab")
else:
    text_query = st.sidebar.text_input("Describe the fabric", "woven twill fabric, stripe pattern")

st.sidebar.subheader(f"Attributes ({len(S.ATTRIBUTES)})")
st.sidebar.caption("Checked attributes define an “attribute match” (and the optional filter).")
left, right = st.sidebar.columns(2)
for name, value in (("Select all", True), ("Clear", False)):
    if (left if value else right).button(name, width="stretch"):
        for attr in S.ATTRIBUTES:
            st.session_state[f"attr_{attr}"] = value
boxes = st.sidebar.columns(2)
for n, attr in enumerate(S.ATTRIBUTES):
    st.session_state.setdefault(f"attr_{attr}", True)
    boxes[n % 2].checkbox(attr, key=f"attr_{attr}")
checked = [a for a in S.ATTRIBUTES if st.session_state[f"attr_{a}"]]

k = st.sidebar.slider("Top K", 3, 20, 10)
image_weight = st.sidebar.slider(
    f"“{S.LEVEL_FUSED}”: weight of the image vector", 0.0, 1.0, 0.5, 0.05, key="fuse_w",
    help="fused = w·image + (1−w)·attributes, both unit-length, then renormalised.",
)
as_filter = st.sidebar.checkbox("Only rank images matching all checked attributes", value=False)
run = st.sidebar.button("Run", type="primary", width="stretch")


def build_query() -> dict | None:
    """Turn the sidebar inputs into a query dict (extracting attributes for an upload)."""
    if source == "Text":
        return {"kind": "text", "key": "text:" + text_query, "text": text_query, "attrs": {}}
    if source == "Pick from catalog":
        meta = corpus_for(IMAGE_LABEL).meta[catalog_id]
        attrs = {a: meta.get(a, "") for a in S.ATTRIBUTES}
        return {"kind": "catalog", "key": f"id:{catalog_id}", "image_id": catalog_id, "attrs": attrs}
    if upload is None:
        st.sidebar.warning("Upload an image first.")
        return None
    data = upload.getvalue()
    matched = catalog_match(upload.name, data)
    if matched:
        meta = corpus_for(IMAGE_LABEL).meta[matched]
        attrs = {a: meta.get(a, "") for a in S.ATTRIBUTES}
        return {
            "kind": "catalog", "key": f"id:{matched}", "image_id": matched, "attrs": attrs,
            "from_upload": upload.name,
        }
    folder = Path(tempfile.mkdtemp(prefix="fabric_ui_"))
    path = folder / f"query{Path(upload.name).suffix.lower()}"
    path.write_bytes(data)
    try:
        with st.spinner("Extracting attributes…"):
            attrs, summary = S.extract_query_attributes(path)
    except Exception as exc:  # no API key / quota: image level still works, the others cannot
        st.sidebar.warning(
            f"Attribute extraction failed ({type(exc).__name__}); attribute metrics and the "
            "attribute / fused levels are unavailable for this upload."
        )
        attrs, summary = {}, {}
    return {
        "kind": "upload",
        "key": "up:" + hashlib.sha1(data).hexdigest()[:12],
        "path": str(path),
        "attrs": attrs,
        "summary": summary,
    }


if run:
    st.session_state["query"] = build_query()
query = st.session_state.get("query")


# ---- searching ------------------------------------------------------------------------------


def query_vector(label: str) -> tuple[np.ndarray, float]:
    """The query's vector in model `label` and the seconds it took (0 for catalog lookups)."""
    cache = st.session_state.setdefault("vec_cache", {})
    key = (label, query["key"], image_weight if S.MODELS[label].level == S.LEVEL_FUSED else None)
    if key not in cache:
        spec = S.MODELS[label]
        if query["kind"] == "catalog":
            corpus = corpus_for(label)
            cache[key] = (corpus.matrix[corpus.ids.index(query["image_id"])], 0.0)
        elif query["kind"] == "upload":
            path = Path(query["path"])
            if spec.level != S.LEVEL_IMAGE and not query["attrs"]:
                raise ValueError("this level needs the upload's attributes, but extraction failed")

            def embed_upload() -> np.ndarray:
                if spec.level == S.LEVEL_IMAGE:
                    return S.embed_image(spec, path, siglip_model)
                attr = S.embed_attributes(spec, query["attrs"], query["summary"], siglip_model)
                if spec.level == S.LEVEL_ATTRS:
                    return attr
                return S.fuse(S.embed_image(spec, path, siglip_model), attr, image_weight)

            cache[key] = S.timed(embed_upload)
        else:
            embed = S.text_embedder(spec, siglip_model)
            cache[key] = S.timed(lambda: embed([query["text"]])[0])
    return cache[key]


def search(label: str, corpus: S.Corpus, top: int | None = None) -> tuple[list[tuple[str, float]], float]:
    vector, seconds = query_vector(label)
    allowed = None
    if as_filter and query["attrs"]:
        allowed = S.matching_images(corpus, query["attrs"], checked)
    exclude = query.get("image_id")
    return S.top_k(corpus, vector, top or k, exclude=exclude, allowed=allowed), seconds


def try_search(label: str, corpus: S.Corpus, top: int | None = None):
    """`search`, or None after showing why this level cannot answer the query."""
    try:
        return search(label, corpus, top)
    except ValueError as exc:
        st.error(f"{label}: {exc}")
        return None


def cards(results, corpus: S.Corpus, shared: set[str], columns: int = 5) -> None:
    for start in range(0, len(results), columns):
        row = st.columns(columns)
        for slot, (image_id, score) in zip(row, results[start : start + columns], strict=False):
            meta = corpus.meta[image_id]
            image = thumbnail(image_id)
            if image is not None:
                slot.image(image, width="stretch")
            mark = "🟢 " if image_id in shared else ""
            slot.caption(f"{mark}score **{score:.2f}**  \n{identity_markdown(image_id)}")


def show_query() -> None:
    cols = st.columns([1, 3])
    if query["kind"] == "text":
        cols[0].info(query["text"])
    else:
        path = Path(query["path"]) if query["kind"] == "upload" else S.image_path(query["image_id"])
        if path is not None:
            cols[0].image(str(path), caption="Query", width="stretch")
        if query["kind"] == "catalog":
            cols[0].markdown(identity_markdown(query["image_id"]))
        else:
            cols[0].caption("Composition: not in an uploaded photo (catalog metadata only).")
    if query["attrs"]:
        shown = {a: v for a, v in query["attrs"].items() if a in checked}
        cols[1].dataframe(
            pd.DataFrame({"attribute": list(shown), "value": list(shown.values())}),
            hide_index=True,
            height=260,
        )


tab_search, tab_compare, tab_score, tab_human = st.tabs(["Search", "Compare", "Scorecard", "Human evaluation"])
base_names = list(S.BASE_MODELS)


def pick(container, name: str, key: str, model_index: int = 0, level_index: int = 0) -> str:
    """Model + level selectors; returns the key into S.MODELS."""
    model = container.selectbox(f"{name}: model", base_names, index=model_index, key=f"{key}_model")
    level = container.selectbox(f"{name}: level", S.LEVELS, index=level_index, key=f"{key}_level")
    return f"{model} · {level}"

with tab_search:
    label = pick(st, "Embedding", "search")
    if not query:
        st.info("Choose a query in the sidebar and press **Run**.")
    else:
        show_query()
        corpus = corpus_for(label)
        outcome = try_search(label, corpus)
        if outcome:
            results, seconds = outcome
            agreement = S.attribute_agreement(results, corpus, query["attrs"], checked)
            st.caption(
                f"{len(results)} results · embedding {seconds:.2f}s"
                + (f" · attribute agreement {agreement:.2f}" if agreement is not None else "")
            )
            cards(results, corpus, set())


def level_results(results, corpus: S.Corpus, shared: set[str], query_attrs: dict, score_name: str = "cosine") -> None:
    """Top results as image + cosine score + composition + this model's attributes (✓ = same as
    the query's attribute from the same model). `corpus.meta` must hold that model's attributes."""
    for image_id, score in results:
        meta = corpus.meta[image_id]
        img_col, info_col = st.columns([1, 2])
        image = thumbnail(image_id)
        if image is not None:
            img_col.image(image, width="stretch")
        mark = "🟢 " if image_id in shared else ""
        info_col.markdown(f"{mark}{score_name} **{score:.3f}**  \n{identity_markdown(image_id)}")
        rows = []
        for attr in checked:
            wanted = query_attrs.get(attr, "")
            rows.append(
                {
                    "attribute": attr,
                    "value": meta.get(attr, "") or "—",
                    "match": "✓" if wanted not in S.UNKNOWN and meta.get(attr) == wanted else "",
                }
            )
        info_col.dataframe(pd.DataFrame(rows), hide_index=True, height=180)


def query_vectors(base: str) -> tuple[S.QueryVectors, float]:
    """The query's image and attribute vectors in model `base` (attribute may be None) and the seconds taken."""
    cache = st.session_state.setdefault("qvec_cache", {})
    key = (base, query["key"])
    if key not in cache:
        spec = S.MODELS[f"{base} · {S.LEVEL_IMAGE}"]
        if query["kind"] == "catalog":
            corpus = corpus_for(spec.label)
            row = corpus.ids.index(query["image_id"])
            attr = corpus.attr[row] if np.linalg.norm(corpus.attr[row]) > 0 else None
            cache[key] = (S.QueryVectors(corpus.image[row], attr), 0.0)
        elif query["kind"] == "upload":

            def embed_upload() -> S.QueryVectors:
                image = S.embed_image(spec, Path(query["path"]), siglip_model)
                attr = None
                if query["attrs"]:
                    attr = S.embed_attributes(spec, query["attrs"], query["summary"], siglip_model)
                return S.QueryVectors(image, attr)

            start = time.perf_counter()
            vectors = embed_upload()
            cache[key] = (vectors, time.perf_counter() - start)
        else:
            embed = S.text_embedder(spec, siglip_model)
            start = time.perf_counter()
            cache[key] = (S.QueryVectors(embed([query["text"]])[0]), time.perf_counter() - start)
    return cache[key]


def run_method(base: str, level: str, method: str, corpus: S.Corpus, q_attrs: dict, top: int):
    """(results, score name, seconds) of `method` for model `base`, or None after showing why not."""
    try:
        vectors, seconds = query_vectors(base)
        allowed = S.matching_images(corpus, q_attrs, checked) if as_filter and q_attrs else None
        results, name = S.search_level(
            corpus, level, method, vectors, k=top, image_weight=image_weight,
            exclude=query.get("image_id"), allowed=allowed, query_attrs=q_attrs, checked=checked,
        )
        return results, name, seconds
    except ValueError as exc:
        st.error(f"{base}: {exc}")
        return None


with tab_compare:
    st.caption(
        "One level and one search method for both models. Each model retrieves its top-K at that level "
        "and shows its own attributes: Gemini = Gemini labels, FashionSigLIP = zero-shot predictions."
    )
    level = st.selectbox("Level", S.LEVELS, key="cmp_level")
    method = st.selectbox("Search method", S.METHODS[level], key=f"cmp_method_{level}")
    if level != S.LEVEL_FUSED:
        st.caption(
            "Vectors are unit length, so cosine, dot product, L2 and squared L2 rank identically (only the "
            "number shown differs). Manhattan, Mahalanobis and Hamming can rank differently. Distances: lower is better."
        )
    else:
        st.caption(
            f"Methods that mix the two signals use the sidebar weight of the image vector (now {image_weight:.2f}); "
            "RRF and DBSF weight image and attributes equally."
        )
    top = st.slider("Top K", 1, 20, 5, key="cmp_k")
    model_a, model_b = base_names[0], base_names[1]
    label_a, label_b = f"{model_a} · {level}", f"{model_b} · {level}"
    fc_attrs, fc_rds = fashionclip_attributes()
    if not query:
        st.info("Choose a query in the sidebar and press **Run**.")
    elif query["kind"] == "text" and level != S.LEVEL_IMAGE:
        st.info("A text query only works at the Image level; pick an image query for the other levels.")
    else:
        catalog_meta = corpus_for(IMAGE_LABEL).meta
        corpus_a, corpus_b = corpus_for(label_a), corpus_for(label_b)
        common = set(corpus_a.ids) & set(corpus_b.ids)
        corpus_a, corpus_b = corpus_a.restrict(common), corpus_b.restrict(common)
        # model B's metadata = its own predicted attributes, so agreement and ✓ use them
        corpus_b = dataclasses.replace(corpus_b, meta={i: fc_attrs.get(i, {}) for i in corpus_b.ids})

        head_img, head_info, head_attrs = st.columns([1, 1, 3])
        attrs_b: dict[str, str] = {}
        if query["kind"] == "catalog":
            qid = query["image_id"]
            path = S.image_path(qid)
            info = identity_markdown(qid)
            attrs_b = fc_attrs.get(qid, {})
            if query.get("from_upload"):
                info += (
                    f"  \n*Upload “{query['from_upload']}” is this catalog image; its own entry is "
                    "left out of the results.*"
                )
        elif query["kind"] == "upload":
            path = Path(query["path"])
            info = (
                "Uploaded image, not in the catalog  \nrd_number / material_id / composition: n/a  \n"
                "FashionSigLIP attributes come from the CSV, so none exist for it"
            )
        else:
            path, info = None, f"Text query:  \n{query['text']}"
        if path is not None:
            head_img.image(str(path), caption="Query", width="stretch")
        head_info.markdown(info)
        if query["attrs"] or attrs_b:
            head_attrs.dataframe(
                pd.DataFrame(
                    {
                        "attribute": checked,
                        f"{model_a} (Gemini)": [query["attrs"].get(a, "") or "—" for a in checked],
                        f"{model_b}": [attrs_b.get(a, "") or "—" for a in checked] if attrs_b else "n/a",
                    }
                ),
                hide_index=True,
                height=260,
            )

        out_a = run_method(model_a, level, method, corpus_a, query["attrs"], top)
        out_b = run_method(model_b, level, method, corpus_b, attrs_b, top)
        shared = {i for i, _ in out_a[0]} & {i for i, _ in out_b[0]} if out_a and out_b else set()
        col_a, col_b = st.columns(2)
        for column, name, out, corpus_x, q_attrs in (
            (col_a, f"{model_a} · Gemini attributes", out_a, corpus_a, query["attrs"]),
            (col_b, f"{model_b} · predicted attributes", out_b, corpus_b, attrs_b),
        ):
            with column:
                st.subheader(name)
                if out:
                    agree = S.attribute_agreement(out[0], corpus_x, q_attrs, checked)
                    st.caption(
                        f"{level} · {method} · {corpus_x.matrix.shape[1]}-d · embedding {out[2]:.2f}s"
                        + (f" · attribute agreement {agree:.2f}" if agree is not None else "")
                        + ("" if S.higher_is_better(out[1]) else " · lower is better")
                    )
                    level_results(out[0], corpus_x, shared, q_attrs, out[1])
        if out_a and out_b:
            st.caption(
                f"🟢 returned by both models ({len(shared)}/{top}): " + (", ".join(sorted(shared)) or "none")
                + ". Raw scores differ in scale between models."
            )
            previous = S.latest_votes(S.load_votes(VOTES_CSV)).get((query["key"], level, method))
            options = ["A", "B", "tie"]
            vote = st.radio(
                "Which result list is better?", options, horizontal=True,
                index=options.index(previous["vote"]) if previous else None, key=f"vote_{query['key']}_{level}_{method}",
            )
            if previous:
                st.caption(f"Already voted {previous['vote']} on {previous['time']}; saving again replaces it.")
            if st.button("Save vote", disabled=vote is None):
                S.log_vote(
                    VOTES_CSV,
                    {
                        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "query_id": query["key"],
                        "query_kind": query["kind"],
                        "level": level,
                        "method": method,
                        "k": str(top),
                        "model_a": label_a,
                        "model_b": label_b,
                        "results_a": "|".join(i for i, _ in out_a[0]),
                        "results_b": "|".join(i for i, _ in out_b[0]),
                        "overlap": str(len(shared)),
                        "checked_attributes": "|".join(checked),
                        "vote": vote or "",
                    },
                )
                st.success(f"Saved to {VOTES_CSV.name}")
                st.rerun()


@st.cache_data(show_spinner="Scoring…")
def score_table(n_queries: int, seed: int, k_score: int, weight: float):
    """Retrieval metrics of every model / level on random catalog queries (cached per setting)."""
    labels_all = list(S.MODELS)
    corpora = {label: corpus_for(label) for label in labels_all}
    pool = sorted(set.intersection(*[set(S.eligible_queries(c)) for c in corpora.values()]))
    chosen = random.Random(seed).sample(pool, min(n_queries, len(pool)))
    table = pd.DataFrame(
        {label: S.retrieval_metrics(corpora[label], chosen, k_score) for label in labels_all}
    ).T
    table.insert(0, "level", [S.MODELS[label].level for label in table.index])
    return table, len(chosen), len(pool)


with tab_score:
    st.caption(
        "Which model retrieves better? Random catalog images are used as queries (each excluded from "
        "its own results). A result is **relevant if it has the same composition as the query**, which "
        "is catalog data neither model produced. Precision@k = share of the k results that are "
        "relevant; Hit@k = share of queries with at least one relevant result; MRR@k = mean 1/rank of "
        "the first relevant result."
    )
    n_queries = st.slider("Queries", 50, 1000, 300, 50, key="score_n")
    seed = st.number_input("Random seed", 0, 9999, 13, key="score_seed")
    k_score = st.slider("k", 1, 20, 5, key="score_k")
    table, n_chosen, n_pool = score_table(n_queries, int(seed), k_score, image_weight)
    st.caption(f"{n_chosen} queries from {n_pool} images whose composition is shared by another image.")
    st.dataframe(table.style.format({c: "{:.3f}" for c in table.columns[1:]}), width="stretch")
    st.bar_chart(table.drop(columns="level"), horizontal=True)


def load_in_compare(vote: dict[str, str]) -> None:
    """Callback: make a saved vote's query the active one and set Compare to its level and method."""
    if vote["query_kind"] == "catalog":
        image_id = vote["query_id"].removeprefix("id:")
        meta = corpus_for(IMAGE_LABEL).meta[image_id]
        st.session_state["query"] = {
            "kind": "catalog", "key": vote["query_id"], "image_id": image_id,
            "attrs": {a: meta.get(a, "") for a in S.ATTRIBUTES},
        }
    st.session_state["cmp_level"] = vote["level"]
    st.session_state[f"cmp_method_{vote['level']}"] = vote["method"]


def result_strip(ids: list[str], other: set[str]) -> None:
    for start in range(0, len(ids), 5):
        for slot, image_id in zip(st.columns(5), ids[start : start + 5], strict=False):
            image = thumbnail(image_id)
            if image is not None:
                slot.image(image, width="stretch")
            slot.caption(("🟢 " if image_id in other else "") + identity_markdown(image_id))


with tab_human:
    st.caption(
        "Every saved vote with the query and the two result lists exactly as they were shown. "
        "Pick one to review it, change the vote, or reload it in Compare."
    )
    saved = S.latest_votes(S.load_votes(VOTES_CSV))
    if not saved:
        st.info("No votes yet. Vote on the Compare tab and it appears here.")
    else:
        votes = sorted(saved.values(), key=lambda v: v["time"], reverse=True)
        counts = pd.Series([v["vote"] for v in votes]).value_counts()
        st.markdown("  ·  ".join(f"**{name}**: {counts.get(name, 0)}" for name in ("A", "B", "tie")) + f"  ·  {len(votes)} voted")
        st.dataframe(
            pd.DataFrame(votes)[["time", "query_id", "level", "method", "k", "overlap", "vote"]],
            hide_index=True, width="stretch",
        )
        def vote_name(v):
            return f"{v['query_id']} · {v['level']} · {v['method']} · voted {v['vote']}"

        chosen = st.selectbox("Review a vote", votes, format_func=vote_name, key="human_pick")
        ids_a, ids_b = chosen["results_a"].split("|"), chosen["results_b"].split("|")
        head, body = st.columns([1, 4])
        if chosen["query_kind"] == "catalog":
            image_id = chosen["query_id"].removeprefix("id:")
            path = S.image_path(image_id)
            if path is not None:
                head.image(str(path), caption="Input image", width="stretch")
            head.markdown(identity_markdown(image_id))
        else:
            head.info(f"{chosen['query_kind']} query ({chosen['query_id']}); its image is not stored.")
        with body:
            st.markdown(f"**A** = {chosen['model_a']}")
            result_strip(ids_a, set(ids_b))
            st.markdown(f"**B** = {chosen['model_b']}")
            result_strip(ids_b, set(ids_a))
        options = ["A", "B", "tie"]
        new_vote = st.radio(
            "Vote", options, index=options.index(chosen["vote"]), horizontal=True,
            key=f"human_vote_{vote_name(chosen)}",
        )
        left, right = st.columns(2)
        if left.button("Save changed vote", disabled=new_vote == chosen["vote"]):
            S.log_vote(
                VOTES_CSV,
                {**chosen, "time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "vote": new_vote},
            )
            st.rerun()
        right.button("Load in Compare", on_click=load_in_compare, args=(chosen,))
