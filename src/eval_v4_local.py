from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from run_pipeline import CACHE, DATA, QCOLS, build_history_candidates, read_parquet
from run_v4 import build_behavior_expansion, predict_microcats, retrieve_augmented


def add_unique(target: list[str], seen: set[str], values: list[str], limit: int) -> None:
    for value in values:
        value = str(value)
        if value not in seen:
            target.append(value)
            seen.add(value)
        if len(target) >= limit:
            break


def select_candidates(
    queries: pd.DataFrame,
    items: pd.DataFrame,
    rankings: list[list[int]],
    predicted_microcats: list[list[str]],
    history: list[list[str]],
    history_quota: int,
    local_quota: int,
    routed_quota: int,
) -> list[list[str]]:
    item_ids = items["item_id"].astype(str).to_numpy()
    item_microcats = items["item_microcat_id"].astype(str).to_numpy()
    item_locations = items["item_location_id"].astype(str).to_numpy()
    answers: list[list[str]] = []
    for row_number, (ranking, microcats, history_ids) in enumerate(
        zip(rankings, predicted_microcats, history)
    ):
        answer: list[str] = []
        seen: set[str] = set()
        add_unique(answer, seen, history_ids, history_quota)

        location = str(queries.iloc[row_number]["search_location_id"])
        microcat_set = set(microcats[:3])
        routed_local = [
            str(item_ids[index])
            for index in ranking
            if item_microcats[index] in microcat_set
            and item_locations[index] == location
        ]
        add_unique(answer, seen, routed_local, history_quota + local_quota)

        routed = [
            str(item_ids[index])
            for index in ranking
            if item_microcats[index] in microcat_set
        ]
        add_unique(
            answer,
            seen,
            routed,
            history_quota + local_quota + routed_quota,
        )
        add_unique(answer, seen, [str(item_ids[index]) for index in ranking], 50)
        answers.append(answer)
    return answers


def recall(answers: list[list[str]], truth: list[set[str]]) -> float:
    return float(
        np.mean(
            [
                len(set(answer[:50]) & positives) / len(positives)
                for answer, positives in zip(answers, truth)
            ]
        )
    )


def ranking_recall(
    rankings: list[list[int]],
    item_ids: np.ndarray,
    truth: list[set[str]],
    topn: int,
) -> float:
    answers = [
        [str(item_ids[index]) for index in ranking[:topn]] for ranking in rankings
    ]
    return recall(answers, truth)


def prepare_split(train: pd.DataFrame, max_queries: int) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, list[set[str]]
]:
    signatures = pd.util.hash_pandas_object(train[QCOLS], index=False).astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.18, random_state=42)
    train_indices, validation_indices = next(
        splitter.split(train, groups=signatures)
    )
    fit_rows = train.iloc[train_indices].copy()
    validation = train.iloc[validation_indices].copy()
    validation["_signature"] = signatures.iloc[validation_indices].to_numpy()

    eval_queries = validation.drop_duplicates("_signature")
    if len(eval_queries) > max_queries:
        eval_queries = eval_queries.sample(max_queries, random_state=42)
    eval_queries = eval_queries[["_signature"] + QCOLS].reset_index(drop=True)
    eval_queries.insert(
        0, "query_id", [f"v4_local_{index:04d}" for index in range(len(eval_queries))]
    )
    truth_by_signature = validation.groupby("_signature")["item_id"].agg(
        lambda values: set(values.astype(str))
    )
    truth = [truth_by_signature[value] for value in eval_queries["_signature"]]
    eval_items = train.drop_duplicates("item_id").reset_index(drop=True)
    return fit_rows, eval_queries, eval_items, truth


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=1000)
    args = parser.parse_args()
    started = time.time()
    CACHE.mkdir(exist_ok=True)
    Path("artifacts").mkdir(exist_ok=True)
    cache_tag = f"v4_local_{args.queries}_seed42"
    ranking_path = CACHE / f"{cache_tag}_rankings.json"
    microcat_path = CACHE / f"{cache_tag}_microcats.json"
    history_path = CACHE / f"{cache_tag}_history.json"

    print("load train and prepare group holdout", flush=True)
    train = read_parquet(DATA / "train.parquet")
    fit_rows, queries, items, truth = prepare_split(train, args.queries)
    print(f"local corpus: {len(queries)} queries, {len(items)} items", flush=True)

    if microcat_path.exists():
        predicted_microcats = json.loads(microcat_path.read_text(encoding="utf-8"))
    else:
        predicted_microcats = predict_microcats(fit_rows, queries)
        microcat_path.write_text(
            json.dumps(predicted_microcats, ensure_ascii=False), encoding="utf-8"
        )

    if ranking_path.exists():
        rankings = json.loads(ranking_path.read_text(encoding="utf-8"))
    else:
        behavior = build_behavior_expansion(fit_rows, items)
        rankings = retrieve_augmented(items, queries, behavior, predicted_microcats)
        ranking_path.write_text(json.dumps(rankings), encoding="utf-8")

    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
    else:
        history = build_history_candidates(
            fit_rows, queries, set(items["item_id"].astype(str))
        )
        history_path.write_text(json.dumps(history, ensure_ascii=False), encoding="utf-8")

    item_ids = items["item_id"].astype(str).to_numpy()
    raw_top50 = ranking_recall(rankings, item_ids, truth, 50)
    oracle_top300 = ranking_recall(rankings, item_ids, truth, 300)

    results: list[dict[str, float | int]] = []
    for history_quota in [0, 4, 8, 12, 16]:
        for local_quota in [0, 8, 16, 24, 32]:
            for routed_quota in [0, 8, 16, 24, 32]:
                if history_quota + local_quota + routed_quota > 48:
                    continue
                answers = select_candidates(
                    queries,
                    items,
                    rankings,
                    predicted_microcats,
                    history,
                    history_quota,
                    local_quota,
                    routed_quota,
                )
                results.append(
                    {
                        "history": history_quota,
                        "local": local_quota,
                        "routed": routed_quota,
                        "recall_at_50": recall(answers, truth),
                    }
                )
    results.sort(key=lambda row: float(row["recall_at_50"]), reverse=True)
    output = {
        "queries": len(queries),
        "items": len(items),
        "raw_augmented_recall_at_50": raw_top50,
        "augmented_oracle_recall_at_300": oracle_top300,
        "best_quota_results": results[:10],
        "seconds": round(time.time() - started, 2),
    }
    Path("artifacts/v4_local_eval.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
