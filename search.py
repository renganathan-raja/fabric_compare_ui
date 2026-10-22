"""Search and comparison logic for the embedding-model comparison UI (no Streamlit in here).

Browsing and scoring read only the deploy/ bundle (utils/prepare_deploy.py builds it). Uploads and
text queries (LIVE mode) reuse the fabric_discovery repo: Gemini client and FashionSigLIP wrapper.
"""

import csv
import importlib.util
import os
import sys
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:  # small-matrix BLAS calls are much slower when spread over every core
    from threadpoolctl import threadpool_limits
except ImportError:  # optional: only a speed-up
    from contextlib import nullcontext

    def threadpool_limits(limits=None, **_):
        return nullcontext()

csv.field_size_limit(sys.maxsize)

ROOT = Path(__file__).resolve().parent
DEPLOY = ROOT / "deploy"  # built by utils/prepare_deploy.py; everything the app reads at runtime
IMAGES_DIR = Path(os.environ["FABRIC_IMAGES_DIR"]) if os.environ.get("FABRIC_IMAGES_DIR") else None
# uploads / text queries need the fabric_discovery repo (Gemini key, torch); the hosted app has neither
LIVE = importlib.util.find_spec("fabric_discovery") is not None
ATTRIBUTES = [
    "color_effect", "pattern", "pattern_scale", "shine_effect", "shine_level", "sheerness",
    "apparent_material_class", "visible_construction", "openness", "structure_visibility", "surface",
    "texture_scale", "yarn_texture", "surface_uniformity", "washed_look", "pile", "surface_fuzz",
    "embellishment",
]
UNKNOWN = ("unsure", "")


LEVEL_IMAGE, LEVEL_ATTRS, LEVEL_FUSED = "Image", "Attributes", "Image + attributes"
LEVELS = (LEVEL_IMAGE, LEVEL_ATTRS, LEVEL_FUSED)


@dataclass(frozen=True)
class ModelSpec:
    """One embedding model at one comparison level.

    `master` is the .npz in deploy/ holding that model's vectors. `level` picks what is searched:
    the image vectors, the attribute-text vectors, or their weighted sum.
    """

    label: str
    master: str
    kind: str  # "gemini" | "siglip"
    level: str


BASE_MODELS = {
    "Gemini Embedding 2 (1536)": ("gemini.npz", "gemini"),
    "FashionSigLIP (768)": ("siglip.npz", "siglip"),
}
MODELS: dict[str, ModelSpec] = {
    f"{name} · {level}": ModelSpec(f"{name} · {level}", master, kind, level)
    for name, (master, kind) in BASE_MODELS.items()
    for level in LEVELS
}


@dataclass
class Corpus:
    """Distinct images of one master: ids, unit vectors (rows) and per-image metadata.

    `matrix` is the vector set of the model level; `image` / `attr` are the image and attribute
    vectors of the same rows (the fusion search methods need both). `cache` holds derived data
    (covariance whitening, binarisation means).
    """

    ids: list[str]
    matrix: np.ndarray
    meta: dict[str, dict[str, str]]
    image: np.ndarray | None = None
    attr: np.ndarray | None = None
    cache: dict = field(default_factory=dict)

    def restrict(self, keep: Collection[str]) -> "Corpus":
        rows = [i for i, image_id in enumerate(self.ids) if image_id in keep]
        if len(rows) == len(self.ids):
            return self  # nothing dropped: keep the object (and its cached whitening) as is
        return Corpus(
            [self.ids[i] for i in rows],
            self.matrix[rows],
            self.meta,
            None if self.image is None else self.image[rows],
            None if self.attr is None else self.attr[rows],
        )


def normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.where(norms == 0, 1, norms)


def fuse(image: np.ndarray, attribute: np.ndarray, image_weight: float) -> np.ndarray:
    """Unit vector of `image_weight`·image + (1 − image_weight)·attribute (both made unit first)."""
    return normalise(image_weight * normalise(image) + (1 - image_weight) * normalise(attribute))


def load_catalog() -> dict[str, dict[str, str]]:
    """image_id -> {rd_number, material_id, composition, the 18 Gemini attributes} from deploy/."""
    with (DEPLOY / "catalog.csv").open(encoding="utf-8", newline="") as handle:
        return {row["image_id"]: row for row in csv.DictReader(handle)}


def load_corpus(spec: ModelSpec, image_weight: float = 0.5) -> Corpus:
    """Images of `spec`'s .npz that have every vector its level needs."""
    data = np.load(DEPLOY / spec.master)
    catalog = load_catalog()
    ids = [str(i) for i in data["ids"]]
    image, attr = data["image"].astype(np.float32), data["attr"].astype(np.float32)
    keep = {
        LEVEL_IMAGE: data["has_image"],
        LEVEL_ATTRS: data["has_attr"],
        LEVEL_FUSED: data["has_image"] & data["has_attr"],
    }[spec.level]
    rows = np.flatnonzero(keep)
    if spec.level == LEVEL_IMAGE:
        matrix = image[rows]
    elif spec.level == LEVEL_ATTRS:
        matrix = attr[rows]
    else:
        matrix = np.stack([fuse(image[i], attr[i], image_weight) for i in rows])
    return Corpus([ids[i] for i in rows], normalise(matrix), catalog, image[rows], attr[rows])


def top_k(
    corpus: Corpus,
    query: np.ndarray,
    k: int,
    *,
    exclude: str | None = None,
    allowed: Collection[str] | None = None,
) -> list[tuple[str, float]]:
    """The k most cosine-similar images to `query` as (image_id, score)."""
    scores = corpus.matrix @ normalise(query.reshape(1, -1))[0]
    if exclude in corpus.meta:
        scores[corpus.ids.index(exclude)] = -np.inf
    if allowed is not None:
        scores[[i for i, image_id in enumerate(corpus.ids) if image_id not in allowed]] = -np.inf
    order = [i for i in np.argsort(-scores, kind="stable") if np.isfinite(scores[i])][:k]
    return [(corpus.ids[i], float(scores[i])) for i in order]


def matching_images(corpus: Corpus, query_attrs: Mapping[str, str], checked: Sequence[str]) -> set[str]:
    """Images whose value equals the query's on every checked attribute (unknown ones ignored)."""
    keys = [a for a in checked if query_attrs.get(a, "") not in UNKNOWN]
    return {
        image_id
        for image_id, row in corpus.meta.items()
        if all(row.get(a) == query_attrs[a] for a in keys)
    }


def attribute_agreement(
    results: Sequence[tuple[str, float]],
    corpus: Corpus,
    query_attrs: Mapping[str, str],
    checked: Sequence[str],
) -> float | None:
    """Mean share of checked attributes each result shares with the query (None if no basis)."""
    keys = [a for a in checked if query_attrs.get(a, "") not in UNKNOWN]
    if not keys or not results:
        return None
    per_result = [
        np.mean([corpus.meta[image_id].get(a) == query_attrs[a] for a in keys])
        for image_id, _ in results
    ]
    return float(np.mean(per_result))


def overlap(a: Sequence[tuple[str, float]], b: Sequence[tuple[str, float]]) -> int:
    return len({i for i, _ in a} & {i for i, _ in b})


def image_path(image_id: str) -> Path | None:
    """The full-size image when FABRIC_IMAGES_DIR is set and has it, else the deploy/ thumbnail."""
    if IMAGES_DIR is not None:
        for suffix in (".jpg", ".jpeg", ".png", ".webp"):
            path = IMAGES_DIR / f"{image_id}{suffix}"
            if path.exists():
                return path
    thumb = DEPLOY / "thumbs" / f"{image_id}.jpg"
    return thumb if thumb.exists() else None


# ---- query embedding (needs the Gemini API key or the local FashionSigLIP model) ----------


def embed_image(spec: ModelSpec, path: Path, siglip: Callable[[], object]) -> np.ndarray:
    """Embed one image file with `spec`. `siglip` supplies a (cached) FashionSigLIP instance."""
    if spec.kind == "siglip":
        return np.array(siglip().embed_images([path])[0], dtype=np.float32)  # type: ignore[attr-defined]
    import mimetypes

    from fabric_discovery.utils.embedding_client import build_client, embed_image_bytes

    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return np.array(embed_image_bytes(build_client(), path.read_bytes(), mime), dtype=np.float32)


def text_embedder(spec: ModelSpec, siglip: Callable[[], object]) -> Callable[[Sequence[str]], np.ndarray]:
    """A function embedding a batch of text queries with the matching text tower."""
    if spec.kind == "siglip":
        model = siglip()
        return lambda texts: np.array(model.embed_texts(list(texts)), dtype=np.float32)  # type: ignore[attr-defined]
    from fabric_discovery.config.settings import settings
    from fabric_discovery.utils.embedding_eval import text_embedder_for

    return text_embedder_for(settings.embedding_model_name, settings.embedding_dimensions)


def extract_query_attributes(path: Path) -> tuple[dict[str, str], dict]:
    """Attributes of an uploaded image from the repo's extraction prompt (one Gemini call).

    Returns (flat per-column strings for display and matching, the raw summary Gemini text-embeds).
    """
    from fabric_discovery.utils.embedding_client import build_client
    from fabric_discovery.utils.fabric_master_builder import (
        attribute_summary,
        extract_attributes,
        flatten_attributes,
    )

    summary = attribute_summary(extract_attributes(build_client(), path))
    return flatten_attributes(summary), summary


def embed_attributes(spec: ModelSpec, flat: Mapping[str, str], summary: Mapping, siglip: Callable[[], object]) -> np.ndarray:
    """Embed a query's attributes the way the master CSV's `attribute_embedding` was built."""
    if spec.kind == "siglip":
        from fabric_discovery.utils.fashionclip_embedding import attribute_text

        text = attribute_text(flat)
        if not text:
            raise ValueError("no informative attributes to embed")
        return np.array(siglip().embed_texts([text])[0], dtype=np.float32)  # type: ignore[attr-defined]
    from fabric_discovery.utils.embedding_client import build_client, embed_attributes_text

    return np.array(embed_attributes_text(build_client(), summary), dtype=np.float32)


def timed(fn: Callable[[], np.ndarray]) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    value = fn()
    return value, time.perf_counter() - start


def log_vote(path: Path, row: Mapping[str, str]) -> None:
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---- retrieval scorecard -------------------------------------------------------------------


def eligible_queries(corpus: Corpus, key: str = "composition") -> list[str]:
    """Images whose `key` is filled and shared with at least one other image (a hit is possible)."""
    counts: dict[str, int] = {}
    for image_id in corpus.ids:
        label = corpus.meta[image_id].get(key, "")
        counts[label] = counts.get(label, 0) + 1
    return [i for i in corpus.ids if corpus.meta[i].get(key, "") and counts[corpus.meta[i][key]] > 1]


def retrieval_metrics(
    corpus: Corpus, query_ids: Sequence[str], k: int, key: str = "composition"
) -> dict[str, float]:
    """Mean Precision@k, Hit@k and MRR@k when a result is relevant if its `key` equals the query's.

    Each query image is searched against the rest of the corpus (itself excluded).
    """
    labels = [corpus.meta[i].get(key, "") for i in corpus.ids]
    position = {image_id: n for n, image_id in enumerate(corpus.ids)}
    precision, hit, reciprocal = [], [], []
    for image_id in query_ids:
        n = position[image_id]
        scores = corpus.matrix @ corpus.matrix[n]
        scores[n] = -np.inf
        top = np.argsort(-scores, kind="stable")[:k]
        relevant = [labels[j] == labels[n] for j in top]
        precision.append(np.mean(relevant))
        hit.append(float(any(relevant)))
        reciprocal.append(1 / (relevant.index(True) + 1) if any(relevant) else 0.0)
    return {
        f"Precision@{k}": float(np.mean(precision)),
        f"Hit@{k}": float(np.mean(hit)),
        f"MRR@{k}": float(np.mean(reciprocal)),
    }


# ---- zero-shot attribute prediction with FashionSigLIP (shared by utils/generate_attribute.py) --

SOFTMAX_SCALE = 100.0  # cosine similarities of unit vectors are tiny; this sharpens the softmax
MIN_CONFIDENCE = 0.4  # below this the predicted value is left empty


def attribute_vocabulary(rows: Collection[Mapping[str, str]]) -> dict[str, list[str]]:
    """Candidate values per attribute: single values seen in `rows`, without "unsure"."""
    found: dict[str, set[str]] = {a: set() for a in ATTRIBUTES}
    for row in rows:
        for attr in ATTRIBUTES:
            value = row.get(attr, "")
            if value and value != "unsure" and "|" not in value:
                found[attr].add(value)
    return {a: sorted(v) for a, v in found.items() if v}


def attribute_prompt(attr: str, value: str) -> str:
    return f"a fabric with {attr.replace('_', ' ')}: {value.replace('_', ' ')}"


def embed_vocabulary(
    vocab: Mapping[str, Sequence[str]], embed_texts: Callable[[list[str]], np.ndarray]
) -> dict[str, np.ndarray]:
    """One unit text vector per (attribute, value) prompt, grouped by attribute."""
    return {
        attr: normalise(np.asarray(embed_texts([attribute_prompt(attr, v) for v in values]), dtype=np.float32))
        for attr, values in vocab.items()
    }


def predict_attributes(
    image_vector: np.ndarray,
    vocab: Mapping[str, Sequence[str]],
    text_vectors: Mapping[str, np.ndarray],
    min_confidence: float = MIN_CONFIDENCE,
) -> dict[str, str]:
    """Per attribute the value whose prompt is closest to the unit image vector ("" if unsure)."""
    out = {}
    for attr, values in vocab.items():
        logits = SOFTMAX_SCALE * (text_vectors[attr] @ image_vector)
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        best = int(np.argmax(probs))
        out[attr] = values[best] if probs[best] >= min_confidence else ""
    return out


# ---- search methods ------------------------------------------------------------------------
# Vector metrics rank by a similarity (higher = better) or a distance (lower = better). `rank` below is
# always "higher = better"; `shown` is the number to display (the distance itself for distances).

METRIC_COSINE, METRIC_DOT, METRIC_L2, METRIC_L2SQ = (
    "Cosine similarity", "Dot product", "Euclidean distance (L2)", "Squared Euclidean distance",
)
METRIC_L1, METRIC_MAHAL, METRIC_HAMMING = (
    "Manhattan distance (L1)", "Mahalanobis distance", "Hamming distance (binarised embedding)",
)
IMAGE_METHODS = (METRIC_COSINE, METRIC_DOT, METRIC_L2, METRIC_L2SQ, METRIC_L1, METRIC_MAHAL, METRIC_HAMMING)

JACCARD, HAMMING_ATTRS, EXACT = (
    "Jaccard similarity (attribute tags)", "Hamming distance (binary attributes)", "Exact matching / filtering",
)
ATTRIBUTE_METHODS = (
    METRIC_COSINE, METRIC_DOT, METRIC_L2, METRIC_L2SQ, METRIC_L1, METRIC_MAHAL, JACCARD, HAMMING_ATTRS, EXACT,
)

SCORE_FUSION, RRF, DBSF = "Weighted score fusion", "Reciprocal Rank Fusion (RRF)", "Distribution-Based Score Fusion (DBSF)"
FILTER_IMAGE = "Attribute filtering + image embedding search"
_CONCAT, _SHARED = "Weighted concatenation", "Shared-space embedding fusion"
_BASIC = {"cosine": METRIC_COSINE, "dot product": METRIC_DOT, "L2": METRIC_L2}
FUSED_METHODS = (
    SCORE_FUSION, RRF, DBSF,
    *(f"{_CONCAT} + {name}" for name in _BASIC),
    *(f"{_SHARED} + {name}" for name in _BASIC),
    FILTER_IMAGE,
)
METHODS = {LEVEL_IMAGE: IMAGE_METHODS, LEVEL_ATTRS: ATTRIBUTE_METHODS, LEVEL_FUSED: FUSED_METHODS}
RRF_K = 60
SHRINKAGE = 0.3  # Mahalanobis: sample covariance alone is singular (fewer images than dimensions)


@dataclass
class QueryVectors:
    """A query's unit vectors in one model; either may be missing (text query: image only)."""

    image: np.ndarray | None = None
    attr: np.ndarray | None = None


def _unit(vector: np.ndarray) -> np.ndarray:
    return normalise(np.asarray(vector, dtype=np.float32).reshape(1, -1))[0]


def vector_scores(metric: str, matrix: np.ndarray, query: np.ndarray, corpus: Corpus, key: str) -> tuple[np.ndarray, np.ndarray]:
    """(rank, shown) of every row of `matrix` for `query` under a vector metric.

    `key` names the matrix inside `corpus.cache` ("image", "attr", ...) for the Mahalanobis whitening
    and the Hamming binarisation threshold (the column means).
    """
    query = np.asarray(query, dtype=np.float32)
    if metric == METRIC_COSINE:
        scores = normalise(matrix) @ _unit(query)
        return scores, scores
    if metric == METRIC_DOT:
        scores = matrix @ query
        return scores, scores
    if metric == METRIC_L2SQ:
        dist = ((matrix - query) ** 2).sum(axis=1)
        return -dist, dist
    if metric == METRIC_L2:
        dist = np.sqrt(((matrix - query) ** 2).sum(axis=1))
        return -dist, dist
    if metric == METRIC_L1:
        dist = np.abs(matrix - query).sum(axis=1)
        return -dist, dist
    if metric == METRIC_MAHAL:
        # Shrunk covariance Σ = (1−s)·S + s·σ²·I (S = sample covariance, σ² = tr S / d). With fewer images
        # than dimensions S has rank < N, so use its SVD: along the N−1 data directions the variance is
        # (1−s)·sv²/(N−1) + s·σ², along every other direction it is s·σ².
        if ("maha", key) not in corpus.cache:
            centered = matrix - matrix.mean(axis=0)
            with threadpool_limits(limits=4):
                _, sv, vt = np.linalg.svd(centered, full_matrices=False)
            sigma2 = float((sv**2).sum() / (len(matrix) - 1) / matrix.shape[1])
            variance = (1 - SHRINKAGE) * sv**2 / (len(matrix) - 1) + SHRINKAGE * sigma2
            corpus.cache[("maha", key)] = (vt, variance, SHRINKAGE * sigma2)
        vt, variance, floor = corpus.cache[("maha", key)]
        delta = matrix - query
        along = delta @ vt.T
        dist = np.sqrt((along**2 / variance).sum(axis=1) + ((delta**2).sum(axis=1) - (along**2).sum(axis=1)) / floor)
        return -dist, dist
    if metric == METRIC_HAMMING:
        if ("mean", key) not in corpus.cache:
            corpus.cache[("mean", key)] = matrix.mean(axis=0)
        mean = corpus.cache[("mean", key)]
        dist = ((matrix > mean) != (query > mean)).sum(axis=1).astype(np.float64)
        return -dist, dist
    raise ValueError(f"unknown metric: {metric}")


def attribute_tags(row: Mapping[str, str], attrs: Sequence[str]) -> set[str]:
    """The "attribute=value" tags of a row over `attrs` (unknown values skipped, "a|b" split)."""
    return {
        f"{a}={v}"
        for a in attrs
        for v in row.get(a, "").split("|")
        if v and v not in UNKNOWN
    }


def _top(corpus: Corpus, rank: np.ndarray, shown: np.ndarray, k: int, exclude: str | None, allowed: Collection[str] | None):
    rank = rank.astype(np.float64).copy()
    if exclude in corpus.meta and exclude in corpus.ids:
        rank[corpus.ids.index(exclude)] = -np.inf
    if allowed is not None:
        rank[[i for i, image_id in enumerate(corpus.ids) if image_id not in allowed]] = -np.inf
    order = [i for i in np.argsort(-rank, kind="stable") if np.isfinite(rank[i])][:k]
    return [(corpus.ids[i], float(shown[i])) for i in order]


def _minmax(values: np.ndarray) -> np.ndarray:
    span = values.max() - values.min()
    return (values - values.min()) / span if span else np.zeros_like(values)


def _ranks(scores: np.ndarray) -> np.ndarray:
    """1-based rank of every row (1 = highest score)."""
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[np.argsort(-scores, kind="stable")] = np.arange(1, len(scores) + 1)
    return ranks


def _dbsf(values: np.ndarray) -> np.ndarray:
    """Distribution-based normalisation: mean ± 3 standard deviations mapped to [0, 1]."""
    mean, std = values.mean(), values.std()
    if std == 0:
        return np.full_like(values, 0.5)
    return np.clip((values - (mean - 3 * std)) / (6 * std), 0.0, 1.0)


def search_level(
    corpus: Corpus,
    level: str,
    method: str,
    query: QueryVectors,
    *,
    k: int,
    image_weight: float = 0.5,
    exclude: str | None = None,
    allowed: Collection[str] | None = None,
    query_attrs: Mapping[str, str] | None = None,
    checked: Sequence[str] = (),
) -> tuple[list[tuple[str, float]], str]:
    """Top-k images of `corpus` for `query` at `level` with search `method`.

    Returns ([(image_id, shown score)], name of the shown score). `query_attrs` (the query's attribute
    values in the same model's labels) and `checked` (attributes that count) serve the tag, exact and
    filter methods. Raises ValueError when the query lacks what the method needs.
    """
    if method not in METHODS[level]:
        raise ValueError(f"{method!r} is not a {level} method")
    vector_attr_method = method in ATTRIBUTE_METHODS[:6]
    if level == LEVEL_IMAGE and query.image is None:
        raise ValueError("this query has no image vector")
    if level == LEVEL_ATTRS and vector_attr_method and query.attr is None:
        raise ValueError("this query has no attribute vector (its attributes are unknown)")
    if level == LEVEL_FUSED and method != FILTER_IMAGE and (query.image is None or query.attr is None):
        raise ValueError("this level needs both the image and the attribute vector of the query")

    if level == LEVEL_IMAGE:
        rank, shown = vector_scores(method, corpus.image, query.image, corpus, "image")
        return _top(corpus, rank, shown, k, exclude, allowed), _score_name(method)

    if level == LEVEL_ATTRS:
        if method in (JACCARD, HAMMING_ATTRS, EXACT):
            if not query_attrs or not any(query_attrs.get(a, "") not in UNKNOWN for a in checked):
                raise ValueError("this method needs the query's attributes")
            q_tags = attribute_tags(query_attrs, checked)
            tags = [attribute_tags(corpus.meta[i], checked) for i in corpus.ids]
            if method == JACCARD:
                rank = shown = np.array([len(q_tags & t) / len(q_tags | t) if q_tags | t else 0.0 for t in tags])
            elif method == HAMMING_ATTRS:
                shown = np.array([float(len(q_tags ^ t)) for t in tags])
                rank = -shown
            else:  # exact: keep only images matching every known checked attribute, ordered by cosine
                exact = matching_images(corpus, query_attrs, checked)
                allowed = exact if allowed is None else set(allowed) & exact
                if query.attr is not None:
                    rank = shown = normalise(corpus.attr) @ _unit(query.attr)
                else:
                    rank = shown = np.zeros(len(corpus.ids))
                return _top(corpus, rank, shown, k, exclude, allowed), "cosine (within exact matches)"
            return _top(corpus, rank, shown, k, exclude, allowed), _score_name(method)
        rank, shown = vector_scores(method, corpus.attr, query.attr, corpus, "attr")
        return _top(corpus, rank, shown, k, exclude, allowed), _score_name(method)

    # LEVEL_FUSED
    if method == FILTER_IMAGE:
        if query.image is None:
            raise ValueError("this query has no image vector")
        if not query_attrs or not any(query_attrs.get(a, "") not in UNKNOWN for a in checked):
            raise ValueError("this method needs the query's attributes")
        keep = matching_images(corpus, query_attrs, checked)
        allowed = keep if allowed is None else set(allowed) & keep
        scores = normalise(corpus.image) @ _unit(query.image)
        return _top(corpus, scores, scores, k, exclude, allowed), "cosine (within attribute matches)"

    img = normalise(corpus.image)
    att = normalise(corpus.attr)
    q_img, q_att = _unit(query.image), _unit(query.attr)
    if method in (SCORE_FUSION, RRF, DBSF):
        s_img, s_att = img @ q_img, att @ q_att
        if method == SCORE_FUSION:
            fused = image_weight * _minmax(s_img) + (1 - image_weight) * _minmax(s_att)
            name = "fused score"
        elif method == RRF:
            fused = 1 / (RRF_K + _ranks(s_img)) + 1 / (RRF_K + _ranks(s_att))
            name = "RRF score"
        else:
            fused = _dbsf(s_img) + _dbsf(s_att)
            name = "DBSF score"
        return _top(corpus, fused, fused, k, exclude, allowed), name

    base = method.split(" + ", 1)[1]
    metric = _BASIC[base]
    if method.startswith(_CONCAT):
        w_i, w_a = np.sqrt(image_weight), np.sqrt(1 - image_weight)  # squared norm = 1, so cosine = w·cos_img + (1−w)·cos_attr
        matrix = np.hstack([w_i * img, w_a * att])
        vector = np.concatenate([w_i * q_img, w_a * q_att])
    else:
        matrix = fuse_rows(img, att, image_weight)
        vector = fuse(q_img, q_att, image_weight)
    rank, shown = vector_scores(metric, matrix, vector, corpus, f"fused:{method}:{image_weight}")
    return _top(corpus, rank, shown, k, exclude, allowed), _score_name(metric)


def fuse_rows(image: np.ndarray, attribute: np.ndarray, image_weight: float) -> np.ndarray:
    """`fuse` for every row of two matrices."""
    return normalise(image_weight * normalise(image) + (1 - image_weight) * normalise(attribute))


def _score_name(method: str) -> str:
    return {
        METRIC_COSINE: "cosine", METRIC_DOT: "dot product", METRIC_L2: "L2 distance", METRIC_L2SQ: "squared L2",
        METRIC_L1: "L1 distance", METRIC_MAHAL: "Mahalanobis", METRIC_HAMMING: "Hamming bits",
        JACCARD: "Jaccard", HAMMING_ATTRS: "differing tags",
    }.get(method, method)


def higher_is_better(score_name: str) -> bool:
    return not any(word in score_name for word in ("L2", "squared", "L1", "Mahalanobis", "Hamming", "differing"))
