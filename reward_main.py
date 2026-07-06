"""Compatibility entrypoint for running the Reward as An Agent FastAPI app."""

from __future__ import annotations

import uvicorn

from reward_as_agent.app import SETTINGS, app


if __name__ == "__main__":
    uvicorn.run(app, host=SETTINGS.host, port=SETTINGS.port)
