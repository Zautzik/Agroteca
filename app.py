"""Serverless entrypoint for the hosted profile (Vercel loads the top-level `app` from here; see
`[tool.vercel]` in pyproject.toml and DEPLOY.md).

The package lives under src/; putting src on the path here makes the import work whether or not
the platform installs the project itself. Locally, run the API with uvicorn as before
(`uvicorn agroteca.api:app`); this file exists only for the serverless platform.
"""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from agroteca.api import app  # noqa: E402  (path setup must come first)

__all__ = ["app"]
