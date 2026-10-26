import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import search as S  # noqa: E402


def _corpus():
    ids = ["a", "b", "c"]
    matrix = S.normalise(np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32))
    meta = {
        "a": {"pattern": "stripe", "surface": "flat"},
        "b": {"pattern": "stripe", "surface": "ribbed"},
        "c": {"pattern": "none", "surface": "flat"},
    }
    return S.Corpus(ids, matrix, meta)


def test_top_k_orders_by_cosine_and_excludes_query():
    corpus = _corpus()
    assert [i for i, _ in S.top_k(corpus, np.array([1.0, 0.0]), 3)] == ["a", "b", "c"]
    assert [i for i, _ in S.top_k(corpus, np.array([1.0, 0.0]), 3, exclude="a")] == ["b", "c"]


def test_filter_and_agreement_ignore_unknown_query_values():
    corpus = _corpus()
    query = {"pattern": "stripe", "surface": "unsure"}
    assert S.matching_images(corpus, query, ["pattern", "surface"]) == {"a", "b"}
    results = [("a", 1.0), ("c", 0.5)]
    assert S.attribute_agreement(results, corpus, query, ["pattern", "surface"]) == 0.5
    assert S.attribute_agreement(results, corpus, {"pattern": "unsure"}, ["pattern"]) is None


def test_restrict_keeps_rows():
    corpus = _corpus().restrict({"a", "c"})
    assert corpus.ids == ["a", "c"] and corpus.matrix.shape == (2, 2)
    assert S.overlap([("a", 1.0), ("b", 0.9)], [("b", 0.2), ("c", 0.1)]) == 1


def test_fuse_is_unit_length_and_weight_moves_it_between_the_inputs():
    image, attribute = np.array([1.0, 0.0]), np.array([0.0, 2.0])
    assert np.allclose(S.fuse(image, attribute, 1.0), [1.0, 0.0])
    assert np.allclose(S.fuse(image, attribute, 0.0), [0.0, 1.0])
    mixed = S.fuse(image, attribute, 0.5)
    assert np.isclose(np.linalg.norm(mixed), 1.0) and np.allclose(mixed, [2**-0.5, 2**-0.5])


def test_models_cover_every_level_for_both_base_models():
    assert len(S.MODELS) == 6
    assert {spec.level for spec in S.MODELS.values()} == set(S.LEVELS)


def test_retrieval_metrics_uses_label_equality():
    corpus = _corpus()
    corpus.meta["a"]["composition"] = corpus.meta["b"]["composition"] = "cotton"
    corpus.meta["c"]["composition"] = "wool"
    assert S.eligible_queries(corpus) == ["a", "b"]
    got = S.retrieval_metrics(corpus, ["a", "b"], 1)
    assert got == {"Precision@1": 1.0, "Hit@1": 1.0, "MRR@1": 1.0}
    got = S.retrieval_metrics(corpus, ["a"], 2)
    assert got["Precision@2"] == 0.5 and got["MRR@2"] == 1.0


def _vec_corpus():
    """4 images; image / attr vectors chosen so the methods disagree in a known way."""
    ids = ["a", "b", "c", "d"]
    image = S.normalise(np.array([[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32))
    attr = S.normalise(np.array([[1, 0, 0], [0, 1, 0], [1, 0.1, 0], [0, 0, 1]], dtype=np.float32))
    meta = {
        "a": {"pattern": "stripe", "surface": "flat"},
        "b": {"pattern": "stripe", "surface": "ribbed"},
        "c": {"pattern": "none", "surface": "flat"},
        "d": {"pattern": "floral", "surface": "unsure"},
    }
    return S.Corpus(ids, image, meta, image, attr)


def _ids(results):
    return [i for i, _ in results]


def test_vector_metrics_rank_and_report_distances():
    corpus = _vec_corpus()
    q = S.QueryVectors(corpus.image[0])
    cos, name = S.search_level(corpus, S.LEVEL_IMAGE, S.METRIC_COSINE, q, k=2, exclude="a")
    assert _ids(cos) == ["b", "c"] and name == "cosine" and S.higher_is_better(name)
    l2, name = S.search_level(corpus, S.LEVEL_IMAGE, S.METRIC_L2, q, k=2, exclude="a")
    assert _ids(l2) == ["b", "c"] and l2[0][1] < l2[1][1] and not S.higher_is_better(name)
    l1, _ = S.search_level(corpus, S.LEVEL_IMAGE, S.METRIC_L1, q, k=1, exclude="a")
    assert np.isclose(l1[0][1], np.abs(corpus.image[1] - corpus.image[0]).sum())
    ham, _ = S.search_level(corpus, S.LEVEL_IMAGE, S.METRIC_HAMMING, q, k=1, exclude="a")
    assert ham[0][1] >= 0


def test_mahalanobis_matches_the_dense_formula():
    rng = np.random.default_rng(0)
    x, q = rng.normal(size=(20, 50)).astype(np.float32), rng.normal(size=50).astype(np.float32)
    corpus = S.Corpus([str(i) for i in range(20)], x, {})
    _, dist = S.vector_scores(S.METRIC_MAHAL, x, q, corpus, "t")
    cov = np.cov(x, rowvar=False)
    cov = (1 - S.SHRINKAGE) * cov + S.SHRINKAGE * np.trace(cov) / 50 * np.eye(50)
    expected = np.sqrt(np.einsum("ij,jk,ik->i", x - q, np.linalg.inv(cov), x - q))
    assert np.allclose(dist, expected, rtol=1e-3)


def test_attribute_set_methods():
    corpus = _vec_corpus()
    q = S.QueryVectors(corpus.image[0], corpus.attr[0])
    attrs, checked = {"pattern": "stripe", "surface": "flat"}, ["pattern", "surface"]
    jac, _ = S.search_level(corpus, S.LEVEL_ATTRS, S.JACCARD, q, k=3, exclude="a", query_attrs=attrs, checked=checked)
    assert _ids(jac)[:2] == ["b", "c"] and jac[0][1] == jac[1][1] == 1 / 3
    ham, _ = S.search_level(corpus, S.LEVEL_ATTRS, S.HAMMING_ATTRS, q, k=1, exclude="a", query_attrs=attrs, checked=checked)
    assert ham[0][1] == 2  # b and c each differ from the query in two tags
    exact, _ = S.search_level(corpus, S.LEVEL_ATTRS, S.EXACT, q, k=3, query_attrs=attrs, checked=checked)
    assert _ids(exact) == ["a"]
    try:
        S.search_level(corpus, S.LEVEL_ATTRS, S.JACCARD, q, k=3, query_attrs={}, checked=checked)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError without query attributes")


def test_fusion_methods():
    corpus = _vec_corpus()
    q = S.QueryVectors(corpus.image[0], corpus.attr[0])
    for method in S.FUSED_METHODS[:-1]:
        res, _ = S.search_level(corpus, S.LEVEL_FUSED, method, q, k=4, image_weight=0.5)
        assert _ids(res)[0] == "a", method  # the query itself is closest in both spaces
    only_image, _ = S.search_level(corpus, S.LEVEL_FUSED, S.SCORE_FUSION, q, k=2, image_weight=1.0, exclude="a")
    assert _ids(only_image)[0] == "b"
    only_attr, _ = S.search_level(corpus, S.LEVEL_FUSED, S.SCORE_FUSION, q, k=2, image_weight=0.0, exclude="a")
    assert _ids(only_attr)[0] == "c"
    rrf, _ = S.search_level(corpus, S.LEVEL_FUSED, S.RRF, q, k=1)
    assert _ids(rrf) == ["a"] and np.isclose(rrf[0][1], 2 / (S.RRF_K + 1))  # rank 1 in both lists
    flt, _ = S.search_level(
        corpus, S.LEVEL_FUSED, S.FILTER_IMAGE, q, k=4, query_attrs={"pattern": "stripe"}, checked=["pattern"]
    )
    assert set(_ids(flt)) == {"a", "b"} and _ids(flt)[0] == "a"


def test_every_listed_method_runs():
    corpus = _vec_corpus()
    q = S.QueryVectors(corpus.image[0], corpus.attr[0])
    for level, methods in S.METHODS.items():
        for method in methods:
            res, name = S.search_level(
                corpus, level, method, q, k=2, query_attrs={"pattern": "stripe"}, checked=["pattern"]
            )
            assert res and name


def test_votes_roundtrip_and_latest_wins(tmp_path):
    path = tmp_path / "votes.csv"
    assert S.load_votes(path) == []
    base = dict.fromkeys(S.VOTE_FIELDS, "x") | {"query_id": "id:1", "level": "L", "method": "M"}
    S.log_vote(path, base | {"vote": "A", "time": "1"})
    S.log_vote(path, base | {"vote": "B", "time": "2"})
    S.log_vote(path, base | {"query_id": "id:2", "vote": "tie", "time": "3"})
    votes = S.load_votes(path)
    assert len(votes) == 3
    latest = S.latest_votes(votes)
    assert latest[("id:1", "L", "M")]["vote"] == "B"
    assert len(latest) == 2
