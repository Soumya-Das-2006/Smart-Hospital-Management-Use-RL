"""
run.py — Application entry point.

WHY THIS EXISTS: Single entry point for both local development (python run.py)
and production (gunicorn run:app). All configuration is read from environment
variables — nothing is hardcoded here.

Industry standards applied:
  - No secrets in this file — everything comes from .env or real env vars
  - PORT / HOST / DEBUG read from AppConfig (which reads from environment)
  - Gunicorn-compatible: exports `app` at module level
  - Clear startup banner so operators know the server is ready
"""

import logging
import os
import sys

# Ensure the project root is on PYTHONPATH so `core.*` and `app.*` resolve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app
from core.config import AppConfig

logger = logging.getLogger(__name__)

# Module-level `app` — required by gunicorn ("gunicorn run:app")
app = create_app()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smart Hospital Resource Allocator")
    parser.add_argument("--host",  default=AppConfig.HOST,  help=f"Host to bind (default: {AppConfig.HOST})")
    parser.add_argument("--port",  type=int, default=AppConfig.PORT, help=f"Port (default: {AppConfig.PORT})")
    parser.add_argument("--debug", action="store_true", default=AppConfig.DEBUG, help="Enable debug mode")
    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════════════════════╗
║   🏥  Smart Hospital Resource Allocator  v1.0.0          ║
║   OpenEnv Hackathon — Production Build                   ║
╠══════════════════════════════════════════════════════════╣
║   Dashboard : http://localhost:{args.port}/dashboard         ║
║   Admin     : http://localhost:{args.port}/admin             ║
║   API Docs  : http://localhost:{args.port}/docs              ║
║   Health    : http://localhost:{args.port}/health            ║
║   API Base  : http://localhost:{args.port}/api               ║
╠══════════════════════════════════════════════════════════╣
║   Debug     : {str(args.debug):<44}║
║   Sessions  : max {AppConfig.MAX_SESSIONS}, TTL {AppConfig.SESSION_TTL_SECONDS}s{' ' * (30 - len(str(AppConfig.MAX_SESSIONS)) - len(str(AppConfig.SESSION_TTL_SECONDS)))}║
╚══════════════════════════════════════════════════════════╝
    """)

    app.run(host=args.host, port=args.port, debug=args.debug)
