from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'


def run(script: str, *arguments: str) -> None:
    command = [sys.executable, str(SRC / script), *arguments]
    print()
    print('> ' + ' '.join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=str(ROOT / 'answer.csv'))
    args = parser.parse_args()

    output = Path(args.output).resolve()
    artifacts = ROOT / 'artifacts'
    artifacts.mkdir(exist_ok=True)
    query_count = '1000'

    run('run_v4.py', '--output', str(artifacts / 'reproduction_v4.csv'))
    run('run_stratified.py', '--output', str(artifacts / 'reproduction_v6.csv'))
    run('eval_v4_local.py', '--queries', query_count)
    run('eval_stratified_local.py', '--queries', query_count)
    run('run_geo_stratified.py', '--mode', 'local', '--queries', query_count)
    run('run_geo_stratified.py', '--mode', 'benchmark')
    run('run_v8_ranker.py', '--output', str(output))
    run('validate_answer.py', str(output))

    print(f'SHA256 {sha256(output)}  {output}', flush=True)


if __name__ == '__main__':
    main()
