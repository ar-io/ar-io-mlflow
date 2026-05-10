"""MLflow RunContextProvider — auto-injects ar.io metadata tags on every run."""

from importlib.metadata import PackageNotFoundError, version

from mlflow.tracking.context.abstract_context import RunContextProvider


def _plugin_version() -> str:
    """Resolve the installed ario-mlflow package version, or "unknown" if not installed."""
    try:
        return version("ario-mlflow")
    except PackageNotFoundError:
        return "unknown"


class ArioContextProvider(RunContextProvider):
    """Auto-injects ar.io tags on every MLflow run via the entry point."""

    def in_context(self) -> bool:
        return True

    def tags(self) -> dict[str, str]:
        return {
            "ario.enabled": "true",
            "ario.version": _plugin_version(),
        }
