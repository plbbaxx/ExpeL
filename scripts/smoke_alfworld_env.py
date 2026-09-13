#!/usr/bin/env python3
from __future__ import annotations
import json
import os
from pathlib import Path
import yaml
from omegaconf import OmegaConf
from alfworld.agents.environment import get_environment

ROOT = Path(__file__).resolve().parents[1]

def main() -> int:
    cfg = OmegaConf.create(yaml.safe_load((ROOT / 'configs/benchmark/alfworld.yaml').read_text(encoding='utf-8')))
    tasks = json.loads((ROOT / 'data/alfworld/manifests/smoke10.json').read_text(encoding='utf-8'))
    gamefile = tasks[0]['gamefile']
    if not Path(gamefile).exists():
        raise FileNotFoundError(f'ALFWorld game file is missing: {gamefile}; ALFWORLD_DATA={os.environ.get("ALFWORLD_DATA")}')
    # ALFWorld exposes environment implementations through this registry.
    # ``AlfredTWEnv`` is not guaranteed to be re-exported from the package.
    env_cls = get_environment(cfg.env.type)
    main_env = env_cls(cfg, train_eval=cfg.split); main_env.game_files = [gamefile]
    env = main_env.init_env(batch_size=1); observations, infos = env.reset()
    initial = observations[0]; steps = []
    commands = list(infos.get('admissible_commands', [[]])[0])
    initial_action_count = len(commands)
    for _ in range(min(3, len(commands))):
        action = commands[0]
        observations, _, done, infos = env.step([action])
        steps.append({'action': action, 'observation': observations[0], 'won': bool(infos['won'][0]), 'done': bool(done[0])})
        commands = list(infos.get('admissible_commands', [[]])[0])
        if done[0] or not commands: break
    result = {'status': 'PASS', 'gamefile': gamefile, 'initial_observation': initial,
              'initial_admissible_action_count': initial_action_count, 'steps': steps}
    out = ROOT / 'artifacts/expel_qwen3_4b/smoke/environment_smoke.json'; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8'); print(json.dumps(result, indent=2)); return 0
if __name__ == '__main__': raise SystemExit(main())
