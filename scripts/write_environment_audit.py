#!/usr/bin/env python3
from __future__ import annotations
import argparse
import importlib.metadata
import os
import platform
from pathlib import Path
import subprocess

PACKAGES = ['torch', 'transformers', 'vllm', 'sentence-transformers', 'faiss-cpu',
            'alfworld', 'hydra-core', 'openai', 'langchain', 'omegaconf']

def version(name: str) -> str:
    try: return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: return 'NOT_INSTALLED_IN_THIS_ENV'

def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.parent.mkdir(parents=True, exist_ok=True)
    lines = [f'python={platform.python_version()}', f'python_executable={os.sys.executable}',
             f'platform={platform.platform()}', f'ALFWORLD_DATA={os.environ.get("ALFWORLD_DATA", "UNSET")}',
             f'git_commit={subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True).stdout.strip()}']
    lines += [f'{name}={version(name)}' for name in PACKAGES]
    args.output.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(args.output)
    return 0
if __name__ == '__main__': raise SystemExit(main())
