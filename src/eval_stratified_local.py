from __future__ import annotations

import argparse
import json
import time
from itertools import product
from pathlib import Path

import numpy as np

from eval_v4_local import prepare_split, recall
from run_pipeline import CACHE, DATA, location_routes, read_parquet
from run_stratified import build_stratified_candidates


def add_unique(target: list[str], seen: set[str], values: list[str], limit: int) -> None:
    if len(target) >= limit:
        return
    for value in values:
        value = str(value)
        if value not in seen:
            target.append(value)
            seen.add(value)
        if len(target) >= limit:
            return


def ids_from_indices(rows: list[list[int]], item_ids: np.ndarray) -> list[list[str]]:
    return [[str(item_ids[index]) for index in row] for row in rows]


def candidate_recall(channel: list[list[str]], truth: list[set[str]]) -> float:
    return float(
        np.mean(
            [
                len(set(candidates) & positives) / len(positives)
                for candidates, positives in zip(channel, truth)
            ]
        )
    )


def assemble(
    channels: dict[str, list[list[str]]],
    quotas: dict[str, int],
) -> list[list[str]]:
    answers: list[list[str]] = []
    for row_number in range(len(next(iter(channels.values())))):
        answer: list[str] = []
        seen: set[str] = set()
        for name in ["history", "local_top1", "local_other", "geo", "routed", "augmented"]:
            add_unique(
                answer,
                seen,
                channels[name][row_number],
                min(50, len(answer) + quotas.get(name, 0)),
            )
        for name in ["local_top1", "local_other", "geo", "routed", "augmented"]:
            add_unique(answer, seen, channels[name][row_number], 50)
        answers.append(answer)
    return answers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=1000)
    args = parser.parse_args()
    started = time.time()
    CACHE.mkdir(exist_ok=True)
    Path("artifacts").mkdir(exist_ok=True)
    cache_tag = f"v4_local_{args.queries}_seed42"

    train = read_parquet(DATA / "train.parquet")
    fit_rows, queries, items, truth = prepare_split(train, args.queries)
    predicted = json.loads((CACHE / f"{cache_tag}_microcats.json").read_text(encoding="utf-8"))

    local_path = CACHE / f"{cache_tag}_stratified_local.json"
    routed_path = CACHE / f"{cache_tag}_stratified_routed.json"
    if local_path.exists() and routed_path.exists():
        local = json.loads(local_path.read_text(encoding="utf-8"))
        routed = json.loads(routed_path.read_text(encoding="utf-8"))
    else:
        local, routed = build_stratified_candidates(items, queries, predicted)
        local_path.write_text(json.dumps(local), encoding="utf-8")
        routed_path.write_text(json.dumps(routed), encoding="utf-8")

    item_ids = items["item_id"].astype(str).to_numpy()
    item_microcats = items["item_microcat_id"].astype(str).to_numpy()
    item_locations = items["item_location_id"].astype(str).to_numpy()
    history = json.loads((CACHE / f"{cache_tag}_history.json").read_text(encoding="utf-8"))
    augmented_idx = json.loads((CACHE / f"{cache_tag}_rankings.json").read_text(encoding="utf-8"))

    local_top1_idx: list[list[int]] = []
    local_other_idx: list[list[int]] = []
    for row, microcats in zip(local, predicted):
        top1 = str(microcats[0]) if microcats else ""
        local_top1_idx.append([index for index in row if item_microcats[index] == top1])
        local_other_idx.append([index for index in row if item_microcats[index] != top1])

    routes = location_routes(fit_rows)
    geo_idx: list[list[int]] = []
    for row_number, row in enumerate(routed):
        search_location = str(queries.iloc[row_number]["search_location_id"])
        route_key = (search_location, str(queries.iloc[row_number]["search_category"]))
        likely_locations = {
            value for value in routes.get(route_key, []) if value != search_location
        }
        geo_idx.append(
            [index for index in row if item_locations[index] in likely_locations]
        )

    channels = {
        "history": [[str(value) for value in row] for row in history],
        "local_top1": ids_from_indices(local_top1_idx, item_ids),
        "local_other": ids_from_indices(local_other_idx, item_ids),
        "geo": ids_from_indices(geo_idx, item_ids),
        "routed": ids_from_indices(routed, item_ids),
        "augmented": ids_from_indices(augmented_idx, item_ids),
    }

    coverage = {
        name: candidate_recall(rows, truth) for name, rows in channels.items()
    }
    union = [
        list(dict.fromkeys(value for name in channels for value in channels[name][row]))
        for row in range(len(queries))
    ]
    coverage["union"] = candidate_recall(union, truth)

    results: list[dict[str, object]] = []
    for history_q, top1_q, other_q, geo_q, routed_q, augmented_q in product(
        [0, 4, 8, 12],
        [20, 24, 28, 32, 36, 40],
        [0, 4, 8],
        [0, 4, 8],
        [0, 4, 8, 12],
        [0, 4, 8, 12],
    ):
        values = [history_q, top1_q, other_q, geo_q, routed_q, augmented_q]
        if sum(values) > 50:
            continue
        quotas = dict(zip(channels, values))
        answers = assemble(channels, quotas)
        results.append({"recall_at_50": recall(answers, truth), **quotas})
    results.sort(key=lambda row: float(row["recall_at_50"]), reverse=True)

    output = {
        "queries": len(queries),
        "items": len(items),
        "candidate_recall": coverage,
        "best": results[:30],
        "seconds": round(time.time() - started, 2),
    }
    output_path = Path(f"artifacts/stratified_local_{args.queries}.json")
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
