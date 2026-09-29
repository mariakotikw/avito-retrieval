from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from run_pipeline import (
    CACHE,
    DATA,
    build_history_candidates,
    normalize_text,
    read_parquet,
    rrf_merge,
    tfidf_candidates,
)


CACHE_VERSION = "v4_behavioral_1"


def normalized(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.replace("ё", "е", regex=False)
        .str.replace(r"[^0-9a-zа-я]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def fingerprint(values: pd.Series) -> str:
    payload = "\n".join(values.astype(str))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def intent_text(frame: pd.DataFrame) -> pd.Series:
    return (
        normalized(frame["search_query"])
        + " "
        + normalized(frame["search_infm_params_text"])
    ).str.strip()


def build_behavior_expansion(train: pd.DataFrame, items: pd.DataFrame) -> list[str]:
    print("build clicked-query expansion by title and microcategory", flush=True)
    pairs = pd.DataFrame(
        {
            "title": normalized(train["item_title_raw"]),
            "microcat": train["item_microcat_id"].astype(str),
            "query": normalized(train["search_query"]),
        }
    )
    pairs = pairs[(pairs["title"] != "") & (pairs["query"] != "")]
    counts = (
        pairs.groupby(["title", "microcat", "query"], sort=False)
        .size()
        .rename("count")
        .reset_index()
        .sort_values("count", ascending=False)
    )
    top = counts.groupby(["title", "microcat"], sort=False).head(8)
    expansions = top.groupby(["title", "microcat"], sort=False)["query"].agg(" ".join)
    lookup = expansions.to_dict()
    item_titles = normalized(items["item_title_raw"])
    item_microcats = items["item_microcat_id"].astype(str)
    result = [
        lookup.get((title, microcat), "")
        for title, microcat in zip(item_titles, item_microcats)
    ]
    print(f"behavior expansion non-empty: {sum(bool(value) for value in result)}/{len(result)}", flush=True)
    return result


def predict_microcats(
    train: pd.DataFrame,
    queries: pd.DataFrame,
    topk: int = 5,
) -> list[list[str]]:
    print("learn query-to-microcategory routing", flush=True)
    train_intent = intent_text(train)
    query_intent = intent_text(queries)
    counts = pd.DataFrame(
        {
            "intent": train_intent,
            "microcat": train["item_microcat_id"].astype(str),
        }
    )
    counts = (
        counts[counts["intent"] != ""]
        .groupby(["intent", "microcat"], sort=False)
        .size()
        .rename("count")
        .reset_index()
    )
    totals = counts.groupby("intent")["count"].transform("sum")
    counts["probability"] = counts["count"] / totals
    grouped = counts.groupby("intent", sort=False)
    intent_microcats = grouped.apply(
        lambda part: list(zip(part["microcat"], part["probability"])),
        include_groups=False,
    )
    historical_intents = intent_microcats.index.tolist()

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=100_000,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )
    vectorizer.fit(historical_intents + query_intent.tolist())
    history_matrix = vectorizer.transform(historical_intents).astype(np.float32)
    query_matrix = vectorizer.transform(query_intent).astype(np.float32)
    exact_lookup = intent_microcats.to_dict()

    predictions: list[list[str]] = []
    for start in range(0, len(queries), 128):
        dense = (query_matrix[start : start + 128] @ history_matrix.T).toarray()
        for offset, similarities in enumerate(dense):
            text = query_intent.iloc[start + offset]
            votes: dict[str, float] = defaultdict(float)
            for microcat, probability in exact_lookup.get(text, []):
                votes[str(microcat)] += 4.0 * float(probability)

            neighbor_count = min(30, similarities.size)
            neighbors = np.argpartition(similarities, -neighbor_count)[-neighbor_count:]
            neighbors = neighbors[np.argsort(-similarities[neighbors])]
            for neighbor in neighbors:
                similarity = float(similarities[neighbor])
                if similarity < 0.25:
                    break
                for microcat, probability in intent_microcats.iloc[neighbor]:
                    votes[str(microcat)] += similarity**3 * float(probability)
            predictions.append(
                sorted(votes, key=votes.get, reverse=True)[:topk]
            )
        print(f"microcat routing: {min(start + 128, len(queries))}/{len(queries)}", flush=True)
    return predictions


def augmented_texts(
    items: pd.DataFrame,
    queries: pd.DataFrame,
    behavior: list[str],
    predicted_microcats: list[list[str]],
) -> tuple[list[str], list[str], list[str], list[str]]:
    title = items["item_title_raw"].map(normalize_text)
    params = items["item_infm_params_text"].map(normalize_text).str.slice(0, 600)
    description = items["item_description_raw"].map(normalize_text).str.slice(0, 300)
    micro = " micro_" + items["item_microcat_id"].astype(str)
    location = " loc_" + items["item_location_id"].astype(str)
    behavior_series = pd.Series(behavior, index=items.index)
    word_docs = (
        title + " " + title + " " + title + " "
        + params + " " + params + " " + description + " "
        + behavior_series + " " + behavior_series + micro + " " + micro + location
    ).tolist()
    char_docs = (
        title + " " + title + " " + params + " "
        + behavior_series + " " + behavior_series + micro + location
    ).tolist()

    query = queries["search_query"].map(normalize_text)
    query_params = queries["search_infm_params_text"].map(normalize_text)
    query_location = " loc_" + queries["search_location_id"].astype(str)
    routing = pd.Series(
        [" ".join(f"micro_{value}" for value in values) for values in predicted_microcats],
        index=queries.index,
    )
    word_queries = (
        query + " " + query + " " + query + " "
        + query_params + " " + query_params + " "
        + routing + " " + routing + " " + routing
        + query_location + " " + query_location
    ).tolist()
    char_queries = (
        query + " " + query + " " + query_params + " "
        + routing + " " + routing + query_location
    ).tolist()
    return word_docs, char_docs, word_queries, char_queries


def retrieve_augmented(
    items: pd.DataFrame,
    queries: pd.DataFrame,
    behavior: list[str],
    predicted_microcats: list[list[str]],
) -> list[list[int]]:
    print("prepare augmented retrieval texts", flush=True)
    word_docs, char_docs, word_queries, char_queries = augmented_texts(
        items, queries, behavior, predicted_microcats
    )
    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        max_features=120_000,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_df=0.99,
        max_features=90_000,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )
    word = tfidf_candidates(word_docs, word_queries, word_vectorizer, 180)
    del word_docs, word_queries
    gc.collect()
    char = tfidf_candidates(char_docs, char_queries, char_vectorizer, 180)
    return [rrf_merge([w, c], [1.0, 1.1], 300) for w, c in zip(word, char)]


def load_v2(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path, dtype=str)
    return frame.set_index("query_id")["answer"].str.split().to_dict()


def add_unique(target: list[str], seen: set[str], values: list[str], limit: int) -> None:
    if len(target) >= limit:
        return
    for value in values:
        value = str(value)
        if value not in seen:
            target.append(value)
            seen.add(value)
        if len(target) >= limit:
            break


def make_answer(
    queries: pd.DataFrame,
    items: pd.DataFrame,
    augmented: list[list[int]],
    predicted_microcats: list[list[str]],
    history: list[list[str]],
    baseline: dict[str, list[str]],
) -> pd.DataFrame:
    item_ids = items["item_id"].astype(str).to_numpy()
    item_microcats = items["item_microcat_id"].astype(str).to_numpy()
    item_locations = items["item_location_id"].astype(str).to_numpy()
    rows: list[dict[str, str]] = []

    for row_number, (query_id, ranking, microcats, history_ids) in enumerate(
        zip(queries["query_id"].astype(str), augmented, predicted_microcats, history)
    ):
        answer: list[str] = []
        seen: set[str] = set()
        add_unique(answer, seen, history_ids[:8], 8)

        location = str(queries.iloc[row_number]["search_location_id"])
        microcat_set = set(microcats[:3])
        routed_local = [
            str(item_ids[idx])
            for idx in ranking
            if item_microcats[idx] in microcat_set and item_locations[idx] == location
        ]
        add_unique(answer, seen, routed_local, 26)

        routed_global = [
            str(item_ids[idx]) for idx in ranking if item_microcats[idx] in microcat_set
        ]
        add_unique(answer, seen, routed_global, 38)

        add_unique(answer, seen, baseline.get(query_id, []), 48)
        add_unique(answer, seen, [str(item_ids[idx]) for idx in ranking], 50)
        rows.append({"query_id": query_id, "answer": " ".join(answer)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="answer_v4.csv")
    parser.add_argument("--baseline", default="answer_v2.csv")
    args = parser.parse_args()
    started = time.time()
    CACHE.mkdir(exist_ok=True)

    print("load data", flush=True)
    train = read_parquet(DATA / "train.parquet")
    queries = read_parquet(DATA / "benchmark_queries.parquet")
    items = read_parquet(DATA / "benchmark_items.parquet")
    metadata = {
        "version": CACHE_VERSION,
        "items": fingerprint(items["item_id"]),
        "queries": fingerprint(queries["query_id"]),
    }
    meta_path = CACHE / "v4_metadata.json"
    ranking_path = CACHE / "v4_augmented_top300.json"
    microcat_path = CACHE / "v4_predicted_microcats.json"
    history_path = CACHE / "v4_history.json"

    cache_valid = False
    if meta_path.exists():
        cache_valid = json.loads(meta_path.read_text(encoding="utf-8")) == metadata
    elif microcat_path.exists():
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        cache_valid = True

    if cache_valid and microcat_path.exists():
        print("load cached microcat routing", flush=True)
        predicted_microcats = json.loads(microcat_path.read_text(encoding="utf-8"))
    else:
        predicted_microcats = predict_microcats(train, queries)
        microcat_path.write_text(
            json.dumps(predicted_microcats, ensure_ascii=False), encoding="utf-8"
        )

    if cache_valid and ranking_path.exists():
        print("load cached augmented candidates", flush=True)
        augmented = json.loads(ranking_path.read_text(encoding="utf-8"))
    else:
        behavior = build_behavior_expansion(train, items)
        augmented = retrieve_augmented(items, queries, behavior, predicted_microcats)
        ranking_path.write_text(json.dumps(augmented), encoding="utf-8")

    if cache_valid and history_path.exists():
        print("load cached history candidates", flush=True)
        history = json.loads(history_path.read_text(encoding="utf-8"))
    else:
        history = build_history_candidates(
            train, queries, set(items["item_id"].astype(str))
        )
        history_path.write_text(json.dumps(history, ensure_ascii=False), encoding="utf-8")

    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    baseline = load_v2(Path(args.baseline))
    answer = make_answer(
        queries, items, augmented, predicted_microcats, history, baseline
    )
    answer.to_csv(args.output, index=False, encoding="utf-8")
    print(
        json.dumps(
            {
                "output": args.output,
                "rows": len(answer),
                "routed_queries": sum(bool(values) for values in predicted_microcats),
                "history_non_empty": sum(bool(values) for values in history),
                "seconds": round(time.time() - started, 2),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
