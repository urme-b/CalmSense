# Contributing to CalmSense

Thanks for your interest. CalmSense is a research codebase, so the bar is **honest, reproducible
results** over features. Contributions that tighten methodology, add tests, or improve clarity are
especially welcome.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
make install-dev          # editable install + dev tools
make demo                 # smoke-test the full pipeline on synthetic data (no download)
```

## Before you open a PR

```bash
make format               # ruff format + autofix
make lint                 # ruff check
make test                 # pytest (must pass; CI enforces ≥60% coverage on src/)
```

- **Target main** with a focused PR; keep unrelated changes out.
- **Add a test** for any behavior change, methodology changes (leakage, calibration, windowing)
  must come with a guard test in tests/.
- **Never weaken the leakage guarantees.** Imputation, scaling, balancing, and calibration are fit
  *inside* each LOSO fold; if you touch the evaluation path, prove the test subject stays unseen.
- **Don't commit generated artifacts** (data/processed/, results/calibration.json,
  results/personalization.json, figures from synthetic runs). The committed results/ are a fixed
  WESAD snapshot, see [results/README.md](results/README.md).
- **Commit messages:** short and concrete (1 to 3 words describing what changed), e.g. honest readme,
  fix leak, dedup windowing.

## Reporting bugs

Open an issue with the command you ran, the expected vs actual behavior, and your OS/Python version.
For security concerns, see [SECURITY.md](SECURITY.md) instead.
