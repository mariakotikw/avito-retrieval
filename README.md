# Avito Retrieval Bootcamp

Проект строит до 50 релевантных объявлений для каждого benchmark-запроса.

Финальное решение объединяет:

- word и character TF-IDF;
- маршрутизацию запросов в microcategory;
- поиск по точной и исторически связанной географии;
- поведенческие признаки из train;
- LightGBM LambdaRank для выбора итогового top-50.

Локальный Recall@50 финальной модели: `0.873028`.

## Структура

- `data/` — исходные parquet-файлы, исключены из Git;
- `cache/` — рассчитанные кандидаты, исключены из Git;
- `artifacts/` — локальные метрики, исключены из Git;
- `src/` — код подготовки кандидатов, обучения и валидации;
- `answer.csv` — финальный результат.

## Установка

```bash
python -m pip install -r requirements.txt
```

## Полное воспроизведение

```bash
python src/run_v4.py --output answer_v4.csv
python src/run_stratified.py --output answer_v6.csv
python src/eval_v4_local.py --queries 1000
python src/eval_stratified_local.py --queries 1000
python src/run_geo_stratified.py --mode local --queries 1000
python src/run_geo_stratified.py --mode benchmark
python src/run_v8_ranker.py --output answer.csv
python src/validate_answer.py answer.csv
```

На Windows Git Bash команды запускаются с префиксом `PYTHONIOENCODING=utf-8`. В PowerShell используется `$env:PYTHONIOENCODING='utf-8'`.
