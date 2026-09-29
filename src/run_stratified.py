from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from run_pipeline import CACHE, DATA, normalize_text, read_parquet, rrf_merge
from run_v4 import add_unique, load_v2


def top_from_subset(scores: np.ndarray, subset: np.ndarray, topn: int) -> list[int]:
    if subset.size == 0:
        return []
    values = scores[subset]
    count = min(topn, subset.size)
    if count == subset.size:
        order = np.argsort(-values)
    else:
        part = np.argpartition(values, -count)[-count:]
        order = part[np.argsort(-values[part])]
    return subset[order].tolist()


def masked_rankings(
    doc_matrix,
    query_matrix,
    item_microcats: np.ndarray,
    item_locations: np.ndarray,
    query_microcats: list[list[str]],
    query_locations: np.ndarray,
    topn: int = 120,
) -> tuple[list[list[int]], list[list[int]]]:
    local_results: list[list[int]] = []
    routed_results: list[list[int]] = []
    for start in range(0, query_matrix.shape[0], 64):
        dense = (query_matrix[start : start + 64] @ doc_matrix.T).toarray()
        for offset, scores in enumerate(dense):
            row_number = start + offset
            microcat_mask = np.isin(item_microcats, query_microcats[row_number][:3])
            routed_subset = np.flatnonzero(microcat_mask)
            local_subset = np.flatnonzero(
                microcat_mask & (item_locations == query_locations[row_number])
            )
            local_results.append(top_from_subset(scores, local_subset, topn))
            routed_results.append(top_from_subset(scores, routed_subset, topn))
        print(
            f"stratified batches: {min(start + 64, query_matrix.shape[0])}/{query_matrix.shape[0]}",
            flush=True,
        )
    return local_results, routed_results


def build_stratified_candidates(
    items: pd.DataFrame,
    queries: pd.DataFrame,
    predicted_microcats: list[list[str]],
) -> tuple[list[list[int]], list[list[int]]]:
    title = items["item_title_raw"].map(normalize_text)
    params = items["item_infm_params_text"].map(normalize_text).str.slice(0, 600)
    description = items["item_description_raw"].map(normalize_text).str.slice(0, 240)
    word_docs = (title + " " + title + " " + title + " " + params + " " + params + " " + description).tolist()
    char_docs = (title + " " + title + " " + params).tolist()
    query = queries["search_query"].map(normalize_text)
    query_params = queries["search_infm_params_text"].map(normalize_text)
    query_docs = (query + " " + query + " " + query_params).tolist()

    item_microcats = items["item_microcat_id"].astype(str).to_numpy()
    item_locations = items["item_location_id"].astype(str).to_numpy()
    query_locations = queries["search_location_id"].astype(str).to_numpy()
    vectorizers = [
        TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_features=90_000,
            sublinear_tf=True,
            norm="l2",
            dtype=np.float32,
        ),
        TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=80_000,
            sublinear_tf=True,
            norm="l2",
            dtype=np.float32,
        ),
    ]
    doc_sets = [word_docs, char_docs]
    local_channels = []
    routed_channels = []
    for vectorizer, docs in zip(vectorizers, doc_sets):
        print(f"fit stratified {vectorizer.analyzer} tfidf", flush=True)
        vectorizer.fit(docs + query_docs)
        doc_matrix = vectorizer.transform(docs).astype(np.float32)
        query_matrix = vectorizer.transform(query_docs).astype(np.float32)
        local, routed = masked_rankings(
            doc_matrix,
            query_matrix,
            item_microcats,
            item_locations,
            predicted_microcats,
            query_locations,
        )
        local_channels.append(local)
        routed_channels.append(routed)

    local_merged = [
        rrf_merge([word, char], [1.0, 1.1], 200)
        for word, char in zip(local_channels[0], local_channels[1])
    ]
    routed_merged = [
        rrf_merge([word, char], [1.0, 1.1], 200)
        for word, char in zip(routed_channels[0], routed_channels[1])
    ]
    return local_merged, routed_merged


def assemble(
    output: str,
    queries: pd.DataFrame,
    items: pd.DataFrame,
    local: list[list[int]],
    routed: list[list[int]],
    history: list[list[str]],
    baseline: dict[str, list[str]],
    quotas: tuple[int, int, int, int],
) -> None:
    history_quota, local_quota, routed_quota, baseline_quota = quotas
    item_ids = items["item_id"].astype(str).to_numpy()
    rows = []
    for query_id, local_row, routed_row, history_ids in zip(
        queries["query_id"].astype(str), local, routed, history
    ):
        answer: list[str] = []
        seen: set[str] = set()
        add_unique(answer, seen, history_ids, history_quota)
        add_unique(
            answer,
            seen,
            [str(item_ids[index]) for index in local_row],
            history_quota + local_quota,
        )
        add_unique(
            answer,
            seen,
            [str(item_ids[index]) for index in routed_row],
            history_quota + local_quota + routed_quota,
        )
        add_unique(
            answer,
            seen,
            baseline.get(query_id, []),
            history_quota + local_quota + routed_quota + baseline_quota,
        )
        add_unique(answer, seen, [str(item_ids[index]) for index in routed_row], 50)
        rows.append({"query_id": query_id, "answer": " ".join(answer)})
    pd.DataFrame(rows).to_csv(output, index=False, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="answer_v6.csv")
    args = parser.parse_args()
    started = time.time()
    queries = read_parquet(DATA / "benchmark_queries.parquet")
    items = read_parquet(DATA / "benchmark_items.parquet")
    predicted = json.loads(
        (CACHE / "v4_predicted_microcats.json").read_text(encoding="utf-8")
    )
    local_path = CACHE / "stratified_local.json"
    routed_path = CACHE / "stratified_routed.json"
    if local_path.exists() and routed_path.exists():
        local = json.loads(local_path.read_text(encoding="utf-8"))
        routed = json.loads(routed_path.read_text(encoding="utf-8"))
    else:
        local, routed = build_stratified_candidates(items, queries, predicted)
        local_path.write_text(json.dumps(local), encoding="utf-8")
        routed_path.write_text(json.dumps(routed), encoding="utf-8")
    history = json.loads((CACHE / "v4_history.json").read_text(encoding="utf-8"))
    baseline = load_v2(Path("answer_v2.csv"))
    assemble(
        args.output,
        queries,
        items,
        local,
        routed,
        history,
        baseline,
        quotas=(10, 24, 8, 8),
    )
    print(
        json.dumps(
            {
                "output": args.output,
                "rows": len(queries),
                "local_non_empty": sum(bool(row) for row in local),
                "routed_non_empty": sum(bool(row) for row in routed),
                "seconds": round(time.time() - started, 2),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
