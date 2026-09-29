from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from eval_v4_local import prepare_split
from run_pipeline import CACHE, DATA, location_routes, normalize_text, read_parquet, rrf_merge
from run_stratified import top_from_subset


def masked_geo_rankings(
    doc_matrix,
    query_matrix,
    items: pd.DataFrame,
    queries: pd.DataFrame,
    predicted_microcats: list[list[str]],
    routes: dict[tuple[str, str], list[str]],
    topn: int = 150,
) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    item_microcats = items["item_microcat_id"].astype(str).to_numpy()
    item_categories = items["item_category_id"].astype(str).to_numpy()
    item_locations = items["item_location_id"].astype(str).to_numpy()
    query_locations = queries["search_location_id"].astype(str).to_numpy()
    query_categories = queries["search_category"].astype(str).to_numpy()
    micro_location_lookup: dict[tuple[str, str], list[int]] = defaultdict(list)
    category_location_lookup: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, (microcat, category, location) in enumerate(
        zip(item_microcats, item_categories, item_locations)
    ):
        micro_location_lookup[(microcat, location)].append(index)
        category_location_lookup[(category, location)].append(index)
    geo_micro_results: list[list[int]] = []
    category_local_results: list[list[int]] = []
    category_geo_results: list[list[int]] = []

    for start in range(0, query_matrix.shape[0], 64):
        dense = (query_matrix[start : start + 64] @ doc_matrix.T).toarray()
        for offset, scores in enumerate(dense):
            row_number = start + offset
            query_location = query_locations[row_number]
            query_category = query_categories[row_number]
            route_locations = routes.get((query_location, query_category), [query_location])
            geo_micro_subset = np.asarray(
                [
                    index
                    for microcat in predicted_microcats[row_number][:5]
                    for location in route_locations
                    for index in micro_location_lookup.get((str(microcat), location), [])
                ],
                dtype=np.int64,
            )
            category_local_subset = np.asarray(
                category_location_lookup.get((query_category, query_location), []),
                dtype=np.int64,
            )
            category_geo_subset = np.asarray(
                [
                    index
                    for location in route_locations
                    for index in category_location_lookup.get((query_category, location), [])
                ],
                dtype=np.int64,
            )
            geo_micro_results.append(
                top_from_subset(scores, geo_micro_subset, topn)
            )
            category_local_results.append(
                top_from_subset(scores, category_local_subset, topn)
            )
            category_geo_results.append(
                top_from_subset(scores, category_geo_subset, topn)
            )
        print(
            f"geo stratified batches: {min(start + 64, query_matrix.shape[0])}/{query_matrix.shape[0]}",
            flush=True,
        )
    return geo_micro_results, category_local_results, category_geo_results


def build_geo_candidates(
    train: pd.DataFrame,
    items: pd.DataFrame,
    queries: pd.DataFrame,
    predicted_microcats: list[list[str]],
) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    title = items["item_title_raw"].map(normalize_text)
    params = items["item_infm_params_text"].map(normalize_text).str.slice(0, 600)
    description = items["item_description_raw"].map(normalize_text).str.slice(0, 240)
    word_docs = (
        title + " " + title + " " + title + " " + params + " " + params + " " + description
    ).tolist()
    char_docs = (title + " " + title + " " + params).tolist()
    query = queries["search_query"].map(normalize_text)
    query_params = queries["search_infm_params_text"].map(normalize_text)
    query_docs = (query + " " + query + " " + query_params).tolist()
    routes = location_routes(train, topk=16)

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
    output_channels: list[list[list[list[int]]]] = []
    for vectorizer, docs in zip(vectorizers, [word_docs, char_docs]):
        print(f"fit geo {vectorizer.analyzer} tfidf", flush=True)
        vectorizer.fit(docs + query_docs)
        doc_matrix = vectorizer.transform(docs).astype(np.float32)
        query_matrix = vectorizer.transform(query_docs).astype(np.float32)
        output_channels.append(
            list(
                masked_geo_rankings(
                    doc_matrix,
                    query_matrix,
                    items,
                    queries,
                    predicted_microcats,
                    routes,
                )
            )
        )

    merged = []
    for channel_number in range(3):
        merged.append(
            [
                rrf_merge([word, char], [1.0, 1.1], 220)
                for word, char in zip(
                    output_channels[0][channel_number],
                    output_channels[1][channel_number],
                )
            ]
        )
    return merged[0], merged[1], merged[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["local", "benchmark"], required=True)
    parser.add_argument("--queries", type=int, default=1000)
    args = parser.parse_args()
    started = time.time()
    CACHE.mkdir(exist_ok=True)
    train = read_parquet(DATA / "train.parquet")

    if args.mode == "local":
        fit_rows, queries, items, _ = prepare_split(train, args.queries)
        tag = f"v4_local_{args.queries}_seed42"
        predicted = json.loads(
            (CACHE / f"{tag}_microcats.json").read_text(encoding="utf-8")
        )
        route_train = fit_rows
        prefix = f"{tag}_geo"
    else:
        queries = read_parquet(DATA / "benchmark_queries.parquet")
        items = read_parquet(DATA / "benchmark_items.parquet")
        predicted = json.loads(
            (CACHE / "v4_predicted_microcats.json").read_text(encoding="utf-8")
        )
        route_train = train
        prefix = "benchmark_geo"

    paths = [
        CACHE / f"{prefix}_micro.json",
        CACHE / f"{prefix}_category_local.json",
        CACHE / f"{prefix}_category_routes.json",
    ]
    if all(path.exists() for path in paths):
        print("geo caches already exist", flush=True)
        return
    channels = build_geo_candidates(route_train, items, queries, predicted)
    for path, channel in zip(paths, channels):
        path.write_text(json.dumps(channel), encoding="utf-8")
    print(
        json.dumps(
            {"mode": args.mode, "queries": len(queries), "items": len(items), "seconds": round(time.time() - started, 2)},
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
