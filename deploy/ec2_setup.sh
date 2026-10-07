#!/usr/bin/env bash
# OPTIONAL: run CoverWise AI on a small Ubuntu EC2 instance.
# Usage: IMAGE=ghcr.io/<owner>/<repo>:v1.0.0 [SECRET_ID=coverwise/prod AWS_REGION=ap-south-1] bash deploy/ec2_setup.sh
# Prefer an IAM instance role + Secrets Manager (SECRET_ID) over .env files. Open only port 8501 (or 443 behind Nginx) in the security group.
set -euo pipefail
: "${IMAGE:?set IMAGE to the container image to run}"
if ! command -v docker >/dev/null; then sudo apt-get update -y && sudo apt-get install -y docker.io; sudo systemctl enable --now docker; fi
[ -n "${GHCR_USER:-}" ] && echo "${GHCR_TOKEN:?}" | sudo docker login ghcr.io -u "$GHCR_USER" --password-stdin
sudo docker pull "$IMAGE"
sudo docker rm -f coverwise 2>/dev/null || true
sudo docker run -d --name coverwise --restart unless-stopped -p 8501:8501 \
  ${SECRET_ID:+-e COVERWISE_AWS_SECRET_ID="$SECRET_ID"} ${AWS_REGION:+-e AWS_REGION="$AWS_REGION"} \
  -e COVERWISE_HUMAN_REVIEW="${COVERWISE_HUMAN_REVIEW:-0}" -v "$HOME/coverwise-labelled:/app/data/labelled" "$IMAGE"
echo "Started. Health: curl http://localhost:8501/_stcore/health   |   Stop unused instances to avoid cost."
