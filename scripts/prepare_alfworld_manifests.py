#!/usr/bin/env python3
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random

def atomic(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8'); tmp.replace(path)

def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument('--source', type=Path, default=Path('data/alfworld/alfworld_tasks_suffix.json')); p.add_argument('--output-dir', type=Path, default=Path('data/alfworld/manifests')); p.add_argument('--seed', type=int, default=42)
    args=p.parse_args(); rows=json.loads(args.source.read_text(encoding='utf-8'))
    if len(rows) != 134: raise ValueError(f'Expected official 134-task list, got {len(rows)}')
    order=list(range(len(rows))); random.Random(args.seed).shuffle(order)
    for name,size in [('smoke10',10),('pilot30',30),('pilot50',50),('full134',134)]:
        chosen=[rows[i] for i in order[:size]]; atomic(args.output_dir / f'{name}.json', chosen)
        atomic(args.output_dir / f'{name}_manifest.json', {'name':name,'seed':args.seed,'source_count':len(rows),'task_count':size,'source_indices':order[:size], 'task_ids':[row['gamefile'] for row in chosen], 'sha256':hashlib.sha256(json.dumps(chosen,sort_keys=True).encode()).hexdigest()})
    return 0
if __name__ == '__main__': raise SystemExit(main())
