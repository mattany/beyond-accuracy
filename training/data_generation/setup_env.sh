#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install it from https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi

cd "${SCRIPT_DIR}"

echo "Creating uv virtualenv at ${VENV_DIR}"
uv venv "${VENV_DIR}" --python 3.10

echo "Installing dependencies"
uv sync --python "${VENV_DIR}/bin/python"

cat <<EOF

Done.

Activate with:
  source ${VENV_DIR}/bin/activate

Then run from the repo root:
  cd ${REPO_ROOT}
  python -m training.data_generation.gen_batch
  python -m training.data_generation.upload_batch_file
  python -m training.data_generation.merge
EOF
