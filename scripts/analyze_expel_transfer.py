#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pickle
import re

RUNS = {'keep_only':'eval_keep_only', 'experience_only':'eval_experience_only',
        'insights_only':'eval_insights_only', 'full_expel':'eval_full_expel'}

def load(path: Path):
    with path.open('rb') as handle: return pickle.load(handle)[-1]

def trajectory_blocks(path: Path):
    text=path.read_text(encoding='utf-8')
    matches=list(re.finditer(r'(?m)^TASK \d+\s*\nFOLD: \d+, EVAL_IDX: (\d+)\s*$', text))
    return {int(match.group(1)):text[match.end():matches[i+1].start() if i+1<len(matches) else len(text)].strip()
            for i,match in enumerate(matches)}

def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument('--base', type=Path, required=True); args=p.parse_args()
    eval_dir=args.base/'alfworld/expel/eval'; states={name:load(eval_dir/f'{run}.pkl') for name,run in RUNS.items()}
    task_count=len(states['keep_only']['tasks']); success={name:set(map(int,state.get('eval_successes',[]))) for name,state in states.items()}
    blocks={name:trajectory_blocks(eval_dir/f'{run}_true.txt') for name,run in RUNS.items()}
    rows=[]; negatives=[]; full_traces=states['full_expel'].get('retrieval_trace',[])
    by_task={}
    for trace in full_traces: by_task.setdefault(trace.get('current_task'),[]).append(trace)
    for index in range(task_count):
        task=states['keep_only']['tasks'][index]; keep=index in success['keep_only']; full=index in success['full_expel']
        transfer='positive_transfer' if not keep and full else 'negative_transfer' if keep and not full else 'neutral'
        row={'task_index':index,'task_id':task.get('env_kwargs',{}).get('gamefile'),'task':task.get('task'),
             **{f'{name}_success':index in values for name,values in success.items()},'transfer':transfer,
             'steps':{name:len(re.findall(r'(?m)^> (?!think:)',blocks[name].get(index,''))) for name in RUNS},
             'invalid_actions':{name:len(re.findall(r'(?i)invalid action|nothing happens',blocks[name].get(index,''))) for name in RUNS}}
        rows.append(row)
        if transfer=='negative_transfer':
            negatives.append({**row,'full_expel_trajectory':blocks['full_expel'].get(index,''),
                              'retrieval_trace':by_task.get(task.get('task'),[])})
    summary={'task_count':task_count,'accuracy':{name:len(values)/task_count for name,values in success.items()},
             'positive_transfer':sum(x['transfer']=='positive_transfer' for x in rows),
             'negative_transfer':len(negatives),'neutral':sum(x['transfer']=='neutral' for x in rows),
             'average_steps':{name:sum(x['steps'][name] for x in rows)/task_count for name in RUNS},
             'invalid_action_count':{name:sum(x['invalid_actions'][name] for x in rows) for name in RUNS}}
    # Avoid relying on a reviewer: this file is a deterministic paired task audit.
    out=args.base/'paired_results.json'; out.write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8')
    (args.base/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    with (args.base/'negative_transfer_cases.jsonl').open('w',encoding='utf-8') as handle:
        for row in negatives: handle.write(json.dumps(row)+'\n')
    print(json.dumps(summary,indent=2)); return 0
if __name__ == '__main__': raise SystemExit(main())
