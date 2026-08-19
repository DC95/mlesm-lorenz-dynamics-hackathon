#!/bin/bash

# Project-level location shared by the organizer and all participants.
# Override HACKATHON_SHARED_ROOT before sourcing this file when reusing the
# challenge under another JSC project account.
export HACKATHON_ENV_NAME="${HACKATHON_ENV_NAME:-lorenz-hackathon-2026}"
export HACKATHON_SHARED_ROOT="${HACKATHON_SHARED_ROOT:-/p/project1/training2635/mlesm-lorenz-hackathon-2026}"
export HACKATHON_ENV_DIR="${HACKATHON_SHARED_ROOT}/env"
export HACKATHON_RUN_ROOT="${HACKATHON_RUN_ROOT:-/p/scratch/training2635/${USER}/mlesm-lorenz-hackathon-2026}"
export HACKATHON_ACCOUNT="${HACKATHON_ACCOUNT:-training2635}"
export HACKATHON_PARTITION="${HACKATHON_PARTITION:-dc-gpu}"

# Challenge 3 shares these reservations with Challenge 5. Re-select the
# reservation on every activation so a shell kept open overnight cannot retain
# yesterday's value. An organizer can use HACKATHON_RESERVATION_OVERRIDE for a
# deliberate override. Outside the event dates the value is left empty so that
# the login-node preflight stops rather than submitting outside a reservation.
if [[ -n "${HACKATHON_RESERVATION_OVERRIDE:-}" ]]; then
    HACKATHON_RESERVATION="${HACKATHON_RESERVATION_OVERRIDE}"
else
    case "$(date +%F)" in
        2026-08-19)
            HACKATHON_RESERVATION=challenge_3_and_5_day1
            ;;
        2026-08-20)
            HACKATHON_RESERVATION=challenge_3_and_5_day2
            ;;
        2026-08-21)
            HACKATHON_RESERVATION=challenge_3_and_5_day3
            ;;
        *)
            HACKATHON_RESERVATION=
            ;;
    esac
fi
export HACKATHON_RESERVATION
