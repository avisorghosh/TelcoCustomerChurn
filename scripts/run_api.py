"""Thin operator script to launch the FastAPI serving application.

Business logic resides strictly inside src/churn_prediction/api/.
"""

import argparse
import sys

import uvicorn

from churn_prediction.api.config import load_serving_config


def main() -> None:
    """Launch the FastAPI server entry point."""
    parser = argparse.ArgumentParser(
        description="Run local FastAPI inference service for Telco Churn prediction."
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Host address to bind the server to.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port number to bind the server to.",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Path to serving YAML configuration file.",
    )
    args = parser.parse_args()

    config = load_serving_config(args.config_path)
    api_config = config.get("api", {})

    host = args.host or api_config.get("host", "127.0.0.1")
    port = args.port or int(api_config.get("port", 8000))

    print(f"=== Starting Telco Churn Prediction API on http://{host}:{port} ===")
    try:
        uvicorn.run(
            "churn_prediction.api.app:create_app",
            factory=True,
            host=host,
            port=port,
            reload=False,
        )
    except Exception as e:
        print(f"Server execution failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
