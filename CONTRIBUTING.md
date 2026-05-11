# Contributing to `ar-io-mlflow`

Thanks for your interest. This is an alpha project — feedback, bug reports,
and PRs are welcome.

## Setup

```bash
git clone https://github.com/ar-io/ar-io-mlflow.git
cd ar-io-mlflow
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
```

## Running tests

```bash
pytest                                          # full plugin suite
pytest tests/test_plugin_smoke.py               # one file
pytest tests/test_plugin_smoke.py::test_name    # one test
pytest -k wallet                                # by keyword
```

No network or MLflow server required. Tests use `tmp_path` and `monkeypatch`
for filesystem and environment isolation; please follow that pattern in
new tests.

## Pull requests

- Open an issue first for non-trivial changes so we can align on direction.
- Keep PRs focused — one cohesive theme per PR is easier to review.
- Update `CHANGELOG.md` under `[Unreleased]` for any user-visible change.
- New behavior gets a test that pins it.
- Documentation lives in [`README.md`](README.md), [`docs/`](docs/), and
  the docstrings of public APIs (`anchor`, `ArioMlflowClient`,
  `VerifiedModel`, etc.). Keep them in sync when shipping changes.

## What's in scope

- The MLflow plugin itself (`ario_mlflow/`)
- The auditor-shaped verification primitives (`verify_record`, `verify_proof_by_tx`)
- The CLI (`ar-io-mlflow verify run|model|trace`, `ar-io-mlflow audit`)
- Documentation, examples, tests for the above

## What's out of scope (here)

- Demo apps using the plugin live in their own repos. A reference demo
  is at
  [vilenarios/Verifiable-AI-Decision-Records-Demo](https://github.com/vilenarios/Verifiable-AI-Decision-Records-Demo).
- The ar.io network itself, Turbo bundler, ar.io Verify gateway services —
  see the relevant ar.io repos.

## Releases

Versions live in two places that must stay in sync:

- `pyproject.toml` `version`
- `ario_mlflow/__init__.py` `__version__`

A test (`tests/test_plugin_safety.py::test_version_matches_pyproject_toml`)
catches drift.

To cut a release: bump both, update `CHANGELOG.md` (move `[Unreleased]`
to a new `[X.Y.Z]` heading), tag (`git tag vX.Y.Z`), publish a GitHub
Release. The `publish.yml` workflow builds and uploads to PyPI via
trusted publishing.

## License

By contributing, you agree your contributions are licensed under
[Apache-2.0](LICENSE).
