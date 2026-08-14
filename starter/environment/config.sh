#!/bin/bash

# Project-level location shared by the organizer and all participants.
# Override HACKATHON_SHARED_ROOT before sourcing this file when reusing the
# challenge under another JSC project account.
export HACKATHON_ENV_NAME="${HACKATHON_ENV_NAME:-lorenz-hackathon-2026}"
export HACKATHON_SHARED_ROOT="${HACKATHON_SHARED_ROOT:-/p/project1/training2635/mlesm-lorenz-hackathon-2026}"
export HACKATHON_ENV_DIR="${HACKATHON_SHARED_ROOT}/env"
