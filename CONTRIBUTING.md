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

## Adding a new dataset (for cross-dataset transfer)

Use [`src/datasets/non_eeg.py`](src/datasets/non_eeg.py) as the template. A dataset module needs one
function that returns a tidy per-window `DataFrame`:

```python
def build(subjects: Optional[list] = None) -> pd.DataFrame:
    # one row per window, with the shared device-agnostic feature columns
    # plus "subject" and "label" (0 = non-stress, 1 = stress).
    ...
```

Then wire it into `scripts/cross_dataset.py` alongside WESAD and Non-EEG, and add its download to
`scripts/download_data.py` (with a SHA-256, see `data/raw/README.md`). Keep the feature space
*device-agnostic* (HRV/EDA/TEMP/ACC summaries), harmonize labels to the binary stress vs. non-stress
contrast, and remember: a robust leave-one-dataset-out claim needs **≥3 corpora with matched stress
constructs** (see PAPER §4.5).

## Reporting bugs

Open an issue with the command you ran, the expected vs actual behavior, and your OS/Python version.
For security concerns, see [SECURITY.md](SECURITY.md) instead.
