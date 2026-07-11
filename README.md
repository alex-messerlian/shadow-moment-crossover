# adaptive-negativity-rl

Reinforcement-learning research codebase.

> **Status: scaffold only.** This repository currently contains project
> structure and a verified environment. No physics, environment, or agent logic
> is implemented yet.

## Requirements

- Python **3.11+** (developed and verified on 3.12)
- The `torch` / `tensordict` / `torchrl` trio is version-coupled — install from
  the pinned `requirements.txt` to get a known-good, mutually compatible set.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# optional: install the package itself in editable mode
pip install -e .
```

## Verify the environment

```bash
.venv/bin/python check_env.py
```

Prints the Python version, the core stack versions, and hardware acceleration
availability (CUDA / Apple MPS).

## Project structure

```text
adaptive-negativity-rl/
├── anrl/                 # main package
│   ├── physics/          # physics / dynamics models
│   ├── envs/             # RL environments (Gymnasium / TorchRL)
│   ├── agents/           # RL agents and policies
│   └── baselines/        # baseline methods for comparison
├── experiments/          # experiment entry points / configs
├── tests/                # test suite
├── results/              # generated artifacts (git-ignored)
├── check_env.py          # environment sanity check
├── requirements.txt      # pinned, reproducible dependency set
├── pyproject.toml        # package metadata + build config
└── README.md
```

## Notes

- `results/` is kept in the tree via `.gitkeep`; its contents are git-ignored.
- Bumping `torch` requires bumping `torchrl` and `tensordict` to a matching
  release — see the header comment in `requirements.txt`.
