# Security Policy

## Reporting a vulnerability

Please report security issues privately to **urme.emma@gmail.com** rather than opening a public issue.
Include steps to reproduce and the affected version/commit. You can expect an acknowledgement within a
few days.

## Threat model & known risks

CalmSense is research software, not a medical device or production service. Two risks are worth
calling out:

- **Pickle deserialization (WESAD).** WESAD subjects are distributed as Python pickles, and
  `src/data/loader.py` unpickles them (`encoding="latin1"`). Unpickling executes arbitrary code —
  **only load `.pkl` files you downloaded from the official WESAD source or generated yourself.** See
  [data/raw/README.md](data/raw/README.md).
- **Model deserialization.** The API loads a trained model from `joblib`/`onnx`. Only run the model
  artifacts produced by this repo's pipeline; do not load model files from untrusted third parties.

Dependencies are pinned (`requirements.txt`) and audited in CI with `pip-audit`. The prediction API
runs as a non-root user in the container and uses a restricted CORS allow-list.
