# AI Agents for Cosmological Anomaly Detection Test Catalogue

This workspace contains a frozen static catalogue for agent-generated CMB
anomaly tests.

The catalogue preserves agent-generated hypotheses, code, results, and
summaries. Literature comparison text is included as provenance and should be
checked before manuscript use.

## Models

- `Mimo V2.5 Pro` (run_directory): `runs/cmb_test_run_mimo`

## Local preview

```bash
python3 -m http.server 8000 --directory docs
```

Then open `http://localhost:8000`.

## Regenerate

```bash
python3 scripts/build_test_catalogue.py
```

The default builder reads `runs/cmb_test_run_mimo` and labels it `Mimo V2.5 Pro`.
To combine multiple model runs, pass each input with a model label:

```bash
python3 scripts/build_test_catalogue.py \
  --run-dir 'runs/cmb_test_run_mimo::Mimo V2.5 Pro' \
  --run-dir 'runs/another_model_run::Other Model'
```

Single JSON exports are also supported:

```bash
python3 scripts/build_test_catalogue.py \
  --run-dir 'runs/cmb_test_run_mimo::Mimo V2.5 Pro' \
  --json 'runs/model_b_results.json::Other Model'
```

The builder copies each test's figures and raw artifacts into `docs/`, and writes:

- `docs/index.html`: browsable static catalogue.
- `docs/tests/<test-id>/`: permanent detail page for each test.
- `docs/data/tests.json`: machine-readable registry.
- `docs/data/tests.csv`: compact table for audit and downstream analysis.
- `docs/data/raw/`: sanitized per-test JSON records preserving generated fields.
- `docs/data/statistics/`: Planck and simulation statistic arrays.

## Publishing

The `docs/` directory is ready for GitHub Pages, Netlify, Cloudflare Pages, or
any static file host. For GitHub Pages, publish the repository from the `docs/`
folder on the default branch.

## Paper citation snippet

```tex
The full catalogue of agent-proposed tests, including generated hypotheses,
analysis code, numerical results, and diagnostic plots, is provided as an
online resource in the frozen Test Catalogue v1.0.
```
