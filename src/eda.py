from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
QCOLS = [
    "search_query",
    "search_location_id",
    "search_is_delivery_search",
    "search_infm_params_text",
    "search_category",
]


def main() -> None:
    train = pd.read_parquet(ROOT / "data" / "train.parquet", columns=QCOLS + ["item_id"])
    queries = pd.read_parquet(ROOT / "data" / "benchmark_queries.parquet", engine="fastparquet")
    items = pd.read_parquet(ROOT / "data" / "benchmark_items.parquet", columns=["item_id", "item_category_id"])

    train_sig = pd.util.hash_pandas_object(train[QCOLS], index=False)
    query_sig = pd.util.hash_pandas_object(queries[QCOLS], index=False)
    positives = train.groupby(train_sig)["item_id"].nunique()
    report = {
        "train_shape": train.shape,
        "benchmark_queries_shape": queries.shape,
        "benchmark_items_shape": items.shape,
        "train_unique_items": int(train["item_id"].nunique()),
        "benchmark_unique_items": int(items["item_id"].nunique()),
        "train_benchmark_item_overlap": int(len(set(train["item_id"]) & set(items["item_id"]))),
        "train_unique_query_signatures": int(train_sig.nunique()),
        "benchmark_exact_query_signatures_in_train": int(query_sig.isin(set(train_sig)).sum()),
        "positive_items_per_query": positives.describe().to_dict(),
        "benchmark_search_categories": queries["search_category"].value_counts().head(10).to_dict(),
        "benchmark_item_categories": items["item_category_id"].value_counts().head(10).to_dict(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
