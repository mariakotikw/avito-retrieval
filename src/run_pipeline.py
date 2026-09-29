from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE = ROOT / "cache"
ARTIFACTS = ROOT / "artifacts"
QCOLS = [
    "search_query",
    "search_location_id",
    "search_is_delivery_search",
    "search_infm_params_text",
    "search_category",
]


def read_parquet(path: Path) -> pd.DataFrame:
    if path.name == "benchmark_queries.parquet":
        return pd.read_parquet(path, engine="fastparquet")
    return pd.read_parquet(path)


def location_routes(
    train: pd.DataFrame, topk: int = 12
) -> dict[tuple[str, str], list[str]]:
    counts = (
        train.assign(
            _search_location=train["search_location_id"].astype(str),
            _category=train["search_category"].astype(str),
            _item_location=train["item_location_id"].astype(str),
        )
        .groupby(["_search_location", "_category", "_item_location"], sort=False)
        .size()
        .rename("count")
        .reset_index()
        .sort_values("count", ascending=False)
    )
    top = counts.groupby(["_search_location", "_category"], sort=False).head(topk)
    return (
        top.groupby(["_search_location", "_category"], sort=False)["_item_location"]
        .agg(list)
        .to_dict()
    )


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).lower().replace("ё", "е")
    text = re.sub(r"[^0-9a-zа-я]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def query_text(df: pd.DataFrame) -> list[str]:
    q = df["search_query"].map(normalize_text)
    params = df["search_infm_params_text"].map(normalize_text)
    loc = " loc_" + df["search_location_id"].astype(str)
    cat = " cat_" + df["search_category"].astype(str)
    delivery = " delivery_" + df["search_is_delivery_search"].astype(str)
    return (q + " " + q + " " + q + " " + params + loc + cat + delivery).tolist()


def item_text(df: pd.DataFrame) -> list[str]:
    title = df["item_title_raw"].map(normalize_text)
    params = df["item_infm_params_text"].map(normalize_text).str.slice(0, 600)
    desc = df["item_description_raw"].map(normalize_text).str.slice(0, 350)
    cat = " cat_" + df["item_category_id"].astype(str)
    micro = " micro_" + df["item_microcat_id"].astype(str)
    loc = " loc_" + df["item_location_id"].astype(str)
    rating = " rating_" + df["item_rating"].fillna(-1).round().astype(int).astype(str)
    return (
        title
        + " "
        + title
        + " "
        + title
        + " "
        + title
        + " "
        + params
        + " "
        + params
        + " "
        + desc
        + cat
        + micro
        + loc
        + rating
    ).tolist()


def item_char_text(df: pd.DataFrame) -> list[str]:
    title = df["item_title_raw"].map(normalize_text)
    params = df["item_infm_params_text"].map(normalize_text).str.slice(0, 450)
    cat = " cat_" + df["item_category_id"].astype(str)
    micro = " micro_" + df["item_microcat_id"].astype(str)
    return (title + " " + title + " " + title + " " + params + cat + micro).tolist()


def topn_from_scores(scores: sparse.spmatrix, topn: int) -> list[list[int]]:
    dense = scores.toarray()
    result: list[list[int]] = []
    for row in dense:
        if row.size <= topn:
            idx = np.argsort(-row)
        else:
            part = np.argpartition(row, -topn)[-topn:]
            idx = part[np.argsort(-row[part])]
        result.append(idx.tolist())
    return result


def tfidf_candidates(
    docs: list[str],
    queries: list[str],
    vectorizer: TfidfVectorizer,
    topn: int,
    batch_size: int = 128,
) -> list[list[int]]:
    all_texts = docs + queries
    print(f"fit {vectorizer.analyzer} tfidf on {len(all_texts)} texts", flush=True)
    vectorizer.fit(all_texts)
    print(f"transform docs for {vectorizer.analyzer} tfidf", flush=True)
    doc_mat = vectorizer.transform(docs).astype(np.float32)
    query_mat = vectorizer.transform(queries).astype(np.float32)
    out: list[list[int]] = []
    for start in range(0, query_mat.shape[0], batch_size):
        sims = query_mat[start : start + batch_size] @ doc_mat.T
        out.extend(topn_from_scores(sims, topn))
        if start == 0 or (start // batch_size) % 5 == 0:
            print(f"{vectorizer.analyzer} batches: {min(start + batch_size, query_mat.shape[0])}/{query_mat.shape[0]}", flush=True)
    return out


def rrf_merge(rankings: list[list[int]], weights: list[float], limit: int) -> list[int]:
    scores: dict[int, float] = defaultdict(float)
    best_rank: dict[int, int] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, idx in enumerate(ranking):
            scores[idx] += weight / (60.0 + rank)
            best_rank[idx] = min(best_rank.get(idx, 10**9), rank)
    ordered = sorted(scores, key=lambda x: (-scores[x], best_rank[x], x))
    return ordered[:limit]


def content_retrieve(items: pd.DataFrame, queries: pd.DataFrame, topn_each: int = 180) -> list[list[int]]:
    print("prepare item/query texts", flush=True)
    docs = item_text(items)
    char_docs = item_char_text(items)
    qdocs = query_text(queries)
    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 1),
        min_df=2,
        max_df=0.96,
        max_features=80_000,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 4),
        min_df=2,
        max_df=0.98,
        max_features=70_000,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )
    word = tfidf_candidates(docs, qdocs, word_vec, topn_each)
    char = tfidf_candidates(char_docs, qdocs, char_vec, topn_each)
    print("merge content rankings", flush=True)
    return [rrf_merge([w, c], [1.0, 1.12], 300) for w, c in zip(word, char)]


def build_history_candidates(train: pd.DataFrame, queries: pd.DataFrame, item_ids: set[str]) -> list[list[str]]:
    print("prepare history candidates", flush=True)
    hist = train[train["item_id"].isin(item_ids)].copy()
    if hist.empty:
        return [[] for _ in range(len(queries))]

    grouped = hist.groupby(QCOLS, dropna=False)["item_id"].agg(lambda s: list(dict.fromkeys(s)))
    hist_df = grouped.reset_index(name="items")
    hdocs = query_text(hist_df)
    qdocs = query_text(queries)

    exact = {
        tuple(row[col] for col in QCOLS): row["items"]
        for _, row in hist_df.iterrows()
    }
    exact_lists = [exact.get(tuple(row[col] for col in QCOLS), []) for _, row in queries.iterrows()]

    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 4),
        min_df=1,
        max_features=90_000,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )
    vec.fit(hdocs + qdocs)
    hmat = vec.transform(hdocs).astype(np.float32)
    qmat = vec.transform(qdocs).astype(np.float32)

    hist_categories = hist_df["search_category"].to_numpy()
    similar_lists: list[list[str]] = []
    for start in range(0, qmat.shape[0], 128):
        dense = (qmat[start : start + 128] @ hmat.T).toarray()
        for local_idx, score_row in enumerate(dense):
            query = queries.iloc[start + local_idx]
            score_row = np.where(hist_categories == query["search_category"], score_row, 0.0)
            candidate_count = min(30, score_row.size)
            top = np.argpartition(score_row, -candidate_count)[-candidate_count:]
            top = top[np.argsort(-score_row[top])]
            item_scores: dict[str, float] = defaultdict(float)
            for hist_idx in top:
                similarity = float(score_row[hist_idx])
                if similarity < 0.45:
                    break
                for item_id in hist_df.iloc[hist_idx]["items"]:
                    item_scores[str(item_id)] += similarity
            similar_lists.append(
                sorted(item_scores, key=item_scores.get, reverse=True)[:50]
            )
        if start == 0 or (start // 128) % 5 == 0:
            print(f"history batches: {min(start + 128, qmat.shape[0])}/{qmat.shape[0]}", flush=True)

    merged: list[list[str]] = []
    for exact_ids, sim_ids in zip(exact_lists, similar_lists):
        ids = []
        seen = set()
        for item_id in list(exact_ids) + sim_ids:
            if item_id not in seen:
                ids.append(item_id)
                seen.add(item_id)
        merged.append(ids)
    return merged


def make_answer(queries: pd.DataFrame, items: pd.DataFrame, content_idx: list[list[int]], history_ids: list[list[str]]) -> pd.DataFrame:
    item_id_array = items["item_id"].astype(str).to_numpy()
    rows = []
    for qid, idxs, hist in zip(queries["query_id"].astype(str), content_idx, history_ids):
        answer: list[str] = []
        seen: set[str] = set()
        for item_id in hist[:10]:
            if item_id not in seen:
                answer.append(item_id)
                seen.add(item_id)
            if len(answer) >= 50:
                break
        if len(answer) < 50:
            for idx in idxs:
                item_id = str(item_id_array[idx])
                if item_id not in seen:
                    answer.append(item_id)
                    seen.add(item_id)
                if len(answer) >= 50:
                    break
        rows.append({"query_id": qid, "answer": " ".join(answer)})
    return pd.DataFrame(rows)


def recall_at_50(answer: pd.DataFrame, truth: pd.Series) -> float:
    pred = answer.set_index("query_id")["answer"].str.split().to_dict()
    scores = []
    for qid, positives in truth.items():
        positives = set(positives)
        scores.append(len(positives.intersection(pred.get(qid, [])[:50])) / len(positives))
    return float(np.mean(scores))


def local_eval(train: pd.DataFrame, max_queries: int | None = None) -> dict[str, float]:
    qsig = train[QCOLS].astype(str).agg("|".join, axis=1)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.18, random_state=42)
    tr_idx, va_idx = next(splitter.split(train, groups=qsig))
    tr = train.iloc[tr_idx].copy()
    va = train.iloc[va_idx].copy()

    eval_queries = va[QCOLS].drop_duplicates().reset_index(drop=True)
    if max_queries and len(eval_queries) > max_queries:
        eval_queries = eval_queries.sample(max_queries, random_state=42).reset_index(drop=True)
    eval_queries.insert(0, "query_id", [f"local_{i:06d}" for i in range(len(eval_queries))])
    key_to_qid = {
        tuple(row[col] for col in QCOLS): row["query_id"]
        for _, row in eval_queries.iterrows()
    }
    va_keys = [tuple(row[col] for col in QCOLS) for _, row in va.iterrows()]
    keep = [key in key_to_qid for key in va_keys]
    va = va.loc[keep].copy()
    va["query_id"] = [key_to_qid[key] for key, selected in zip(va_keys, keep) if selected]
    truth = va.groupby("query_id")["item_id"].agg(lambda s: list(dict.fromkeys(s)))

    eval_items = train.drop_duplicates("item_id").reset_index(drop=True)
    content_idx = content_retrieve(eval_items, eval_queries, topn_each=100)
    hist_ids = build_history_candidates(tr, eval_queries, set(eval_items["item_id"].astype(str)))
    ans = make_answer(eval_queries, eval_items, content_idx, hist_ids)
    return {
        "queries": float(len(eval_queries)),
        "items": float(len(eval_items)),
        "recall_at_50": recall_at_50(ans, truth),
    }


def final_run(output: Path) -> dict[str, object]:
    print("load parquet files", flush=True)
    train = read_parquet(DATA / "train.parquet")
    queries = read_parquet(DATA / "benchmark_queries.parquet")
    items = read_parquet(DATA / "benchmark_items.parquet")

    content_idx = content_retrieve(items, queries, topn_each=100)
    history_ids = build_history_candidates(train, queries, set(items["item_id"].astype(str)))
    print("write answer", flush=True)
    answer = make_answer(queries, items, content_idx, history_ids)
    answer.to_csv(output, index=False, encoding="utf-8")
    return {
        "output": str(output),
        "rows": len(answer),
        "queries": len(queries),
        "items": len(items),
        "history_non_empty": int(sum(bool(x) for x in history_ids)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["final", "local-eval"], default="final")
    parser.add_argument("--output", default=str(ROOT / "answer.csv"))
    parser.add_argument("--eval-queries", type=int, default=None)
    args = parser.parse_args()

    CACHE.mkdir(exist_ok=True)
    ARTIFACTS.mkdir(exist_ok=True)
    start = time.time()
    if args.mode == "local-eval":
        train = read_parquet(DATA / "train.parquet")
        result = local_eval(train, max_queries=args.eval_queries)
    else:
        result = final_run(Path(args.output))
    result["seconds"] = round(time.time() - start, 2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
