#!/usr/bin/env bash
# Run from the project root
set -e

PROJECT_NAME="electronics-intel"

# Exclude virtualenv, caches, and generated files
zip -r "${PROJECT_NAME}.zip" . \
  -x ".venv/*" \
  -x "**/__pycache__/*" \
  -x "*.pyc" \
  -x ".env" \
  -x "chroma_db/*" \
  -x "models/*" \
  -x "logs/*" \
  -x "data/raw/*" \
  -x "data/processed/*" \
  -x "data/finetune/*" \
  -x "*.zip"

echo "✓ Created ${PROJECT_NAME}.zip"
