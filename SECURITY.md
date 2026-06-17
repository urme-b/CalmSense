# Security Policy

## Reporting a vulnerability

Please report security issues privately to **urme.emma@gmail.com** rather than opening a public issue.
Include steps to reproduce and the affected version/commit. You can expect an acknowledgement within a
few days.

## Threat model & known risks

CalmSense is research software, not a medical device or production service.

- **Pickle deserialization (WESAD).** WESAD subjects are distributed as Python pickles, and
  `src/data/loader.py` unpickles them (`encoding="latin1"`). Unpickling executes arbitrary code —
  **only load `.pkl` files you downloaded from the official WESAD source or generated yourself.** See
  [data/raw/README.md](data/raw/README.md).
- **Model deserialization (trust boundary).** The API loads exactly one model from a fixed path with
  no user-supplied location — `outputs/models/stress_classifier.joblib` (`api/model.py`). `joblib`
  uses pickle, so loading executes code: the server is only safe to run with the model **this repo's
  own pipeline produced** (`scripts/run_experiment.py`). Its SHA-256 is pinned in
  [`outputs/models/stress_classifier.joblib.sha256`](outputs/models/stress_classifier.joblib.sha256);
  verify before serving with `shasum -a 256 -c outputs/models/stress_classifier.joblib.sha256`. Never
  point the loader at a third-party `.joblib`.

## API surface

The prediction service (`api/main.py`) exposes only `GET /health`, `GET /model`, `POST /predict`, and
`POST /explain`. Request bodies are validated by pydantic schemas (`api/schemas.py`); inputs are numeric
feature maps, not files.

- **No file-upload endpoint** and **no WebSocket** exist server-side. The dashboard's CSV picker parses
  client-side in the browser (`frontend/src/services/api.ts`, `parseFloat` only) and never uploads raw
  bytes to the server.
- **CORS** is locked to `http://localhost:3000` and `https://urme-b.github.io` (override via
  `CALMSENSE_CORS_ORIGINS`), methods limited to GET/POST, and `allow_credentials=False` — never a
  wildcard origin paired with credentials.

## Supply chain & secrets

Dependencies are pinned (`requirements.txt`) and audited in CI with `pip-audit`. The prediction API
runs as a non-root user in the container. The full git history (all refs) is scanned for committed
secrets with [gitleaks](https://github.com/gitleaks/gitleaks) — last run: **0 findings**.
