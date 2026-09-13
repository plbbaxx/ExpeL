# ExpeL ALFWorld reproduction with local Qwen3-4B

## Frozen upstream

- Upstream: `https://github.com/LeapLabTHU/ExpeL`
- Commit: `e41ec9a24823e7b560c561ab191441b56d9bcefc`
- Branch: `main`
- Commit date: `2024-12-20T13:28:00+08:00`
- Official Python: 3.9.17

## Invariants

The ExpeL experience gathering, reflection, insight extraction, prompts,
experience representation, FAISS retrieval, task-similarity strategy,
all-mpnet-base-v2 embedder, Top-2 final experiences, maximum 20 ALFWorld
steps, and environment success signal are unchanged.

Engineering adaptations are restricted to:

1. an OpenAI-compatible endpoint for Qwen3-4B;
2. removal of optional `<think>...</think>` text before the original parser;
3. global reproducibility seeds and side-channel retrieval/token audit logs;
4. fixed 10/30/50/134 task manifests and resumable runner scripts.

## Frozen decoding

| field | value |
|---|---:|
| served model | `Qwen3-4B-Instruct-2507` |
| temperature | 0 |
| top_p | 1 |
| max_tokens | 512 |
| seed | 42 |
| stop | original ExpeL `newline` / `double newline` |
| thinking | model checkpoint is used as served; leaked think tags are stripped before parsing |
| vLLM generation config | `vllm` (do not inherit checkpoint sampling defaults) |

## Experiment order

1. `run_alfworld_environment_smoke.sh`
2. `run_react_smoke10.sh`
3. `run_expel_pilot30.sh`
4. `run_expel_pilot30_eval.sh`

Do not proceed from step 2 if environment errors or systematic parser failures
occur. Do not scale past pilot30 until the paired result summary is reviewed.

## Artifact layout

```text
artifacts/expel_qwen3_4b/
  environment_audit.txt
  model_server/
  smoke/
    environment_smoke.json
    react10/
  pilot30/
    alfworld/expel/
    paired_results.json
    summary.json
    negative_transfer_cases.jsonl
```

Official checkpoints remain `.pkl`, `.txt`, and `_true.txt` files under each
run's `alfworld/expel` directory. The runners never create tmux sessions; only
the model-server launcher uses tmux.
