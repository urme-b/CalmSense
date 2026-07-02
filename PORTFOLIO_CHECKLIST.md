# Portfolio checklist

A per-repo bar for research/ML code that has to be defended line by line. Tick each item;
keep the status honest.

| # | Item | CalmSense |
| --- | --- | --- |
| 1 | Clean-clone reproduction works (`pip install -e .` → `make demo` → one prediction) | ✅ verified + CI `clean-clone` job |
| 2 | No secrets in tree or history (gitleaks) | ✅ history scanned, CI gitleaks job |
| 3 | No raw data or licence-restricted files committed | ✅ WESAD gitignored, never in history |
| 4 | Subject-wise evaluation, no leakage | ✅ LOSO grouped by subject, per-fold preprocessing, leakage tests |
| 5 | Seeds set + dependencies pinned | ✅ `set_seed()` + `config.SEED`; pinned reqs + lockfile |
| 6 | No tracked logs/outputs/caches; real `.gitignore` | ✅ only `.gitkeep` + deliberate result figures |
| 7 | Plain-voice README, working commands, real (LOSO) numbers | ✅ no hype/emoji, numbers reconcile with `results/` |
| 8 | De-AI'd code (`ruff`, `vulture` clean) | ✅ vulture clean, ruff clean, mypy clean |
| 9 | Green CI running real tests | ✅ lint + mypy + pytest + pip-audit + gitleaks + clean-clone |
| 10 | Sharp description + 4–6 specific topics | ✅ set via `gh` |
| 11 | Clean lowercase commit style, single author | ✅ 1–2 word messages, `urme-b`, no trailers |

## How to verify each

```bash
# 1 clean clone
python -m venv .venv && . .venv/bin/activate && pip install -e . && make demo
# 2 secrets
gitleaks detect --source . -v
# 4 leakage guard
pytest tests/test_methodology.py -q
# 5 determinism
pytest tests/test_determinism.py -q
# 6 tracked artifacts
git ls-files | grep -iE 'logs/|\.log$|\.pt$|\.ckpt$|__pycache__|ipynb_checkpoints|\.DS_Store'   # -> empty
# 8 de-AI
ruff check src/ && vulture src/ --min-confidence 80 && mypy src/ --ignore-missing-imports
# 9 CI locally
ruff check src/ tests/ scripts/ && pytest -q
```

## Other repos

Run the same bar on each remaining repo in a separate session (each is its own git tree):
`Multimodal`, `Sensor`. A profile reads as "engineer who ships" only when every repo clears the
same bar, not just one.
