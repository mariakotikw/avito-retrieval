from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from eval_v4_local import prepare_split
from run_pipeline import CACHE, DATA, QCOLS, location_routes, normalize_text, read_parquet


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def id_channels_to_indices(
    rows: list[list[str]], item_index: dict[str, int]
) -> list[list[int]]:
    return [
        [item_index[value] for value in map(str, row) if value in item_index]
        for row in rows
    ]


def build_behavior_stats(frame: pd.DataFrame) -> tuple[dict, ...]:
    query = frame["search_query"].map(normalize_text)
    params = frame["search_infm_params_text"].map(normalize_text)
    intent = (query + " " + params).str.strip()
    microcat = frame["item_microcat_id"].astype(str)

    query_counts = (
        pd.DataFrame({"query": query, "microcat": microcat})
        .groupby(["query", "microcat"], sort=False)
        .size()
        .rename("count")
        .reset_index()
    )
    query_counts["total"] = query_counts.groupby("query")["count"].transform("sum")
    query_micro_probability = {
        (row.query, row.microcat): row.count / row.total
        for row in query_counts.itertuples(index=False)
    }
    query_total = query_counts.groupby("query")["total"].first().to_dict()

    intent_counts = (
        pd.DataFrame({"intent": intent, "microcat": microcat})
        .groupby(["intent", "microcat"], sort=False)
        .size()
        .rename("count")
        .reset_index()
    )
    intent_counts["total"] = intent_counts.groupby("intent")["count"].transform("sum")
    intent_micro_probability = {
        (row.intent, row.microcat): row.count / row.total
        for row in intent_counts.itertuples(index=False)
    }
    intent_total = intent_counts.groupby("intent")["total"].first().to_dict()

    location_counts = (
        frame.assign(
            _search_location=frame["search_location_id"].astype(str),
            _category=frame["search_category"].astype(str),
            _item_location=frame["item_location_id"].astype(str),
        )
        .groupby(["_search_location", "_category", "_item_location"], sort=False)
        .size()
        .rename("count")
        .reset_index()
    )
    location_counts["total"] = location_counts.groupby(
        ["_search_location", "_category"]
    )["count"].transform("sum")
    location_probability = {
        (search_location, category, item_location): count / total
        for search_location, category, item_location, count, total
        in location_counts.itertuples(index=False, name=None)
    }
    return (
        query_micro_probability,
        intent_micro_probability,
        location_probability,
        query_total,
        intent_total,
    )


def build_rows(
    query_indices: list[int],
    queries: pd.DataFrame,
    items: pd.DataFrame,
    predicted: list[list[str]],
    channels: dict[str, list[list[int]]],
    popularity: np.ndarray,
    item_texts: tuple[np.ndarray, np.ndarray],
    routes: dict[tuple[str, str], list[str]],
    behavior_stats: tuple[dict, dict, dict, dict, dict],
    truth: list[set[str]] | None,
) -> tuple[np.ndarray, np.ndarray | None, list[int], list[list[int]]]:
    item_ids = items["item_id"].astype(str).to_numpy()
    microcats = items["item_microcat_id"].astype(str).to_numpy()
    locations = items["item_location_id"].astype(str).to_numpy()
    rating = items["item_rating"].fillna(0).to_numpy(dtype=np.float32)
    reviews = np.log1p(
        items["item_rating_reviews_count"].fillna(0).to_numpy(dtype=np.float32)
    )
    price = np.log1p(items["item_price"].fillna(0).clip(lower=0).to_numpy(dtype=np.float32))
    item_categories = items["item_category_id"].astype(str).to_numpy()
    title_texts, parameter_texts = item_texts
    features: list[list[float]] = []
    labels: list[int] = []
    groups: list[int] = []
    candidate_rows: list[list[int]] = []
    names = list(channels)
    (
        query_micro_probability,
        intent_micro_probability,
        location_probability,
        query_total,
        intent_total,
    ) = behavior_stats

    for query_number in query_indices:
        rank_maps = [
            {index: rank for rank, index in enumerate(channels[name][query_number])}
            for name in names
        ]
        candidates = list(
            dict.fromkeys(
                index
                for name in names
                for index in channels[name][query_number]
            )
        )
        candidate_rows.append(candidates)
        groups.append(len(candidates))
        query_location = str(queries.iloc[query_number]["search_location_id"])
        query_category = str(queries.iloc[query_number]["search_category"])
        query_text = normalize_text(queries.iloc[query_number]["search_query"])
        query_parameter_text = normalize_text(
            queries.iloc[query_number]["search_infm_params_text"]
        )
        query_tokens = set(query_text.split())
        query_parameter_tokens = set(query_parameter_text.split())
        intent_text = (query_text + " " + query_parameter_text).strip()
        query_numbers = {token for token in query_tokens if any(char.isdigit() for char in token)}
        route_order = {
            value: rank
            for rank, value in enumerate(routes.get((query_location, query_category), []))
        }
        microcat_order = {
            str(value): rank for rank, value in enumerate(predicted[query_number][:5])
        }
        positives = truth[query_number] if truth is not None else None
        for index in candidates:
            row: list[float] = []
            for rank_map in rank_maps:
                rank = rank_map.get(index)
                row.extend(
                    [
                        0.0 if rank is None else 1.0,
                        0.0 if rank is None else 1.0 / (1.0 + rank),
                        600.0 if rank is None else float(rank),
                    ]
                )
            microcat_rank = microcat_order.get(microcats[index], 3)
            route_rank = route_order.get(locations[index])
            title_tokens = set(title_texts[index].split())
            parameter_tokens = set(parameter_texts[index].split())
            item_tokens = title_tokens | parameter_tokens
            item_numbers = {
                token for token in item_tokens if any(char.isdigit() for char in token)
            }
            query_size = max(1, len(query_tokens))
            parameter_size = max(1, len(query_parameter_tokens))
            union_size = max(1, len(query_tokens | title_tokens))
            number_size = max(1, len(query_numbers))
            row.extend(
                [
                    float(locations[index] == query_location),
                    float(item_categories[index] == query_category),
                    float(microcat_rank == 0),
                    float(microcat_rank == 1),
                    float(microcat_rank == 2),
                    float(microcat_rank == 3),
                    float(microcat_rank == 4),
                    float(query_micro_probability.get((query_text, microcats[index]), 0.0)),
                    float(intent_micro_probability.get((intent_text, microcats[index]), 0.0)),
                    float(
                        location_probability.get(
                            (query_location, query_category, locations[index]), 0.0
                        )
                    ),
                    float(np.log1p(query_total.get(query_text, 0))),
                    float(np.log1p(intent_total.get(intent_text, 0))),
                    0.0 if route_rank is None else 1.0,
                    0.0 if route_rank is None else 1.0 / (1.0 + route_rank),
                    len(query_tokens & title_tokens) / query_size,
                    len(query_tokens & item_tokens) / query_size,
                    len(query_parameter_tokens & parameter_tokens) / parameter_size,
                    len(query_parameter_tokens & item_tokens) / parameter_size,
                    len(query_tokens & title_tokens) / union_size,
                    float(bool(query_tokens) and query_tokens <= title_tokens),
                    float(bool(query_text) and query_text in title_texts[index]),
                    len(query_numbers & item_numbers) / number_size,
                    float(bool(query_numbers) and not query_numbers <= item_numbers),
                    float(rating[index]),
                    float(reviews[index]),
                    float(price[index]),
                    float(popularity[index]),
                ]
            )
            features.append(row)
            if positives is not None:
                labels.append(int(item_ids[index] in positives))
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.int8) if truth is not None else None,
        groups,
        candidate_rows,
    )


def score_recall(
    model: lgb.LGBMRanker,
    x: np.ndarray,
    groups: list[int],
    candidates: list[list[int]],
    item_ids: np.ndarray,
    truth: list[set[str]],
    query_indices: list[int],
) -> float:
    scores = model.predict(x)
    recalls: list[float] = []
    offset = 0
    for group, row, query_number in zip(groups, candidates, query_indices):
        order = np.argsort(-scores[offset : offset + group])[:50]
        selected = {str(item_ids[row[position]]) for position in order}
        recalls.append(len(selected & truth[query_number]) / len(truth[query_number]))
        offset += group
    return float(np.mean(recalls))


def fit_ranker(x: np.ndarray, y: np.ndarray, groups: list[int]) -> lgb.LGBMRanker:
    model = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=180,
        learning_rate=0.045,
        num_leaves=31,
        max_depth=8,
        min_child_samples=40,
        reg_lambda=1.0,
        verbosity=-1,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x, y, group=groups)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--output", default="answer_v8.csv")
    args = parser.parse_args()
    started = time.time()
    tag = "v4_local_1000_seed42"
    train = read_parquet(DATA / "train.parquet")
    fit_rows, local_queries, local_items, truth = prepare_split(train, 1000)
    local_predicted = load_json(CACHE / f"{tag}_microcats.json")
    local_index = {
        value: index for index, value in enumerate(local_items["item_id"].astype(str))
    }
    local_history = load_json(CACHE / f"{tag}_history.json")
    local_channels = {
        "history": id_channels_to_indices(local_history, local_index),
        "local": load_json(CACHE / f"{tag}_stratified_local.json"),
        "routed": load_json(CACHE / f"{tag}_stratified_routed.json"),
        "augmented": load_json(CACHE / f"{tag}_rankings.json"),
    }
    local_geo_paths = {
        "geo_micro": CACHE / f"{tag}_geo_micro.json",
        "category_local": CACHE / f"{tag}_geo_category_local.json",
        "category_routes": CACHE / f"{tag}_geo_category_routes.json",
    }
    for name, path in local_geo_paths.items():
        if path.exists():
            local_channels[name] = load_json(path)
    local_counts = fit_rows.groupby("item_id").size().to_dict()
    local_popularity = np.log1p(
        local_items["item_id"].map(local_counts).fillna(0).to_numpy(dtype=np.float32)
    )
    local_texts = (
        local_items["item_title_raw"].map(normalize_text).to_numpy(),
        local_items["item_infm_params_text"].map(normalize_text).to_numpy(),
    )
    local_routes = location_routes(fit_rows, topk=16)
    local_behavior = build_behavior_stats(fit_rows)

    rng = np.random.default_rng(42)
    shuffled = rng.permutation(len(local_queries))
    train_queries = shuffled[:800].tolist()
    valid_queries = shuffled[800:].tolist()
    x_train, y_train, train_groups, _ = build_rows(
        train_queries, local_queries, local_items, local_predicted,
        local_channels, local_popularity, local_texts, local_routes,
        local_behavior, truth
    )
    x_valid, _, valid_groups, valid_candidates = build_rows(
        valid_queries, local_queries, local_items, local_predicted,
        local_channels, local_popularity, local_texts, local_routes,
        local_behavior, truth
    )
    model = fit_ranker(x_train, y_train, train_groups)
    validation_recall = score_recall(
        model, x_valid, valid_groups, valid_candidates,
        local_items["item_id"].astype(str).to_numpy(), truth, valid_queries
    )
    print(
        f"heldout local recall@50: {validation_recall:.6f}; channels={list(local_channels)}",
        flush=True,
    )
    if args.local_only:
        return

    del x_train, y_train, train_groups, x_valid, valid_groups, valid_candidates, model
    gc.collect()

    all_queries = list(range(len(local_queries)))
    x_all, y_all, all_groups, _ = build_rows(
        all_queries, local_queries, local_items, local_predicted,
        local_channels, local_popularity, local_texts, local_routes,
        local_behavior, truth
    )
    model = fit_ranker(x_all, y_all, all_groups)
    del x_all, y_all, all_groups
    gc.collect()

    queries = read_parquet(DATA / "benchmark_queries.parquet")
    items = read_parquet(DATA / "benchmark_items.parquet")
    predicted = load_json(CACHE / "v4_predicted_microcats.json")
    item_index = {value: index for index, value in enumerate(items["item_id"].astype(str))}
    history = load_json(CACHE / "v4_history.json")
    channels = {
        "history": id_channels_to_indices(history, item_index),
        "local": load_json(CACHE / "stratified_local.json"),
        "routed": load_json(CACHE / "stratified_routed.json"),
        "augmented": load_json(CACHE / "v4_augmented_top300.json"),
    }
    benchmark_geo_paths = {
        "geo_micro": CACHE / "benchmark_geo_micro.json",
        "category_local": CACHE / "benchmark_geo_category_local.json",
        "category_routes": CACHE / "benchmark_geo_category_routes.json",
    }
    for name, path in benchmark_geo_paths.items():
        if path.exists():
            channels[name] = load_json(path)
    if list(channels) != list(local_channels):
        raise RuntimeError(
            f"local/benchmark channel mismatch: {list(local_channels)} != {list(channels)}"
        )
    counts = train.groupby("item_id").size().to_dict()
    popularity = np.log1p(
        items["item_id"].map(counts).fillna(0).to_numpy(dtype=np.float32)
    )
    benchmark_texts = (
        items["item_title_raw"].map(normalize_text).to_numpy(),
        items["item_infm_params_text"].map(normalize_text).to_numpy(),
    )
    benchmark_routes = location_routes(train, topk=16)
    benchmark_behavior = build_behavior_stats(train)
    benchmark_queries = list(range(len(queries)))
    x_benchmark, _, benchmark_groups, benchmark_candidates = build_rows(
        benchmark_queries, queries, items, predicted, channels, popularity,
        benchmark_texts, benchmark_routes, benchmark_behavior, None
    )
    scores = model.predict(x_benchmark)
    item_ids = items["item_id"].astype(str).to_numpy()
    train_signatures = set(pd.util.hash_pandas_object(train[QCOLS], index=False).to_numpy())
    query_signatures = pd.util.hash_pandas_object(queries[QCOLS], index=False).to_numpy()
    exact_history = np.fromiter(
        (value in train_signatures for value in query_signatures),
        dtype=bool,
        count=len(queries),
    )
    rows = []
    offset = 0
    for row_number, (query_id, group, candidates) in enumerate(zip(
        queries["query_id"].astype(str), benchmark_groups, benchmark_candidates
    )):
        order = np.argsort(-scores[offset : offset + group])
        selected: list[int] = []
        seen: set[int] = set()
        if exact_history[row_number]:
            for index in channels["history"][row_number][:10]:
                if index not in seen:
                    selected.append(index)
                    seen.add(index)
        for position in order:
            index = candidates[position]
            if index not in seen:
                selected.append(index)
                seen.add(index)
            if len(selected) >= 50:
                break
        answer = " ".join(str(item_ids[index]) for index in selected)
        rows.append({"query_id": query_id, "answer": answer})
        offset += group
    pd.DataFrame(rows).to_csv(args.output, index=False, encoding="utf-8")
    print(
        json.dumps(
            {
                "output": args.output,
                "heldout_local_recall_at_50": validation_recall,
                "seconds": round(time.time() - started, 2),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
