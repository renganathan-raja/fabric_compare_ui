#!/usr/bin/env bash
# Start the comparison UI. The venv's .pth file makes the fabric_discovery repo importable.
cd "$(dirname "$0")" && exec .venv/bin/streamlit run app.py "$@"
