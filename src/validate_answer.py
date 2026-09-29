from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def read_queries() -> pd.DataFrame:
    return pd.read_parquet(ROOT / "data" / "benchmark_queries.parquet", engine="fastparquet")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("answer", nargs="?", default=str(ROOT / "answer.csv"))
    args = parser.parse_args()

    answer_path = Path(args.answer)
    answer = pd.read_csv(answer_path, dtype=str).fillna("")
    queries = read_queries()
    items = pd.read_parquet(ROOT / "data" / "benchmark_items.parquet")

    assert list(answer.columns) == ["query_id", "answer"], answer.columns.tolist()
    assert len(answer) == len(queries), (len(answer), len(queries))
    assert answer["query_id"].is_unique
    assert set(answer["query_id"]) == set(queries["query_id"].astype(str))

    item_set = set(items["item_id"].astype(str))
    bad_counts = []
    bad_items = []
    duplicated = []
    for qid, text in zip(answer["query_id"], answer["answer"]):
        ids = text.split()
        if len(ids) > 50 or len(ids) == 0:
            bad_counts.append((qid, len(ids)))
        if len(ids) != len(set(ids)):
            duplicated.append(qid)
        missing = [x for x in ids if x not in item_set]
        if missing:
            bad_items.append((qid, missing[:3]))

    assert not bad_counts, bad_counts[:5]
    assert not duplicated, duplicated[:5]
    assert not bad_items, bad_items[:5]
    print(f"OK: {answer_path} has {len(answer)} rows and valid top-50 item_id lists.")


if __name__ == "__main__":
    main()
