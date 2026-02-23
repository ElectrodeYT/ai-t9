#!/usr/bin/env bash
# Build the ai-t9 Docker trainer image.
#
# Usage:
#   bash scripts/build_image.sh [TAG]
#
# TAG defaults to "ai-t9-trainer:latest".
#
# After building, push to your registry before using on Vast.ai:
#   docker push <registry>/<user>/ai-t9-trainer:latest
#
# Then run via orchestrator:
#   python scripts/vast_orchestrate.py configs/vast-large.yaml \
#       --image <registry>/<user>/ai-t9-trainer:latest --install skip

set -euo pipefail

TAG="${1:-ai-t9-trainer:latest}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "Building image: ${TAG}"
echo "Context:        ${REPO_ROOT}"
echo ""

docker build \
    --tag "${TAG}" \
    --progress=plain \
    "${REPO_ROOT}"

echo ""
echo "Done. Image tagged as: ${TAG}"
echo ""
echo "To use with vast_orchestrate.py, push it to a registry first:"
echo "  docker tag ${TAG} <registry>/<user>/${TAG}"
echo "  docker push <registry>/<user>/${TAG}"
echo ""
echo "Then pass --image to the orchestrator:"
echo "  python scripts/vast_orchestrate.py configs/vast-large.yaml \\"
echo "      --image <registry>/<user>/${TAG} --install skip"
