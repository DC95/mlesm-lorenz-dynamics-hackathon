#!/bin/bash
set -euo pipefail

for required_name in \
    HACKATHON_SHARED_ROOT \
    HACKATHON_ENV_DIR \
    HACKATHON_RUN_ROOT \
    HACKATHON_ACCOUNT \
    HACKATHON_PARTITION \
    HACKATHON_RESERVATION
do
    if [[ -z "${!required_name:-}" ]]; then
        echo "${required_name} is unset. First run: source environment/activate.sh" >&2
        exit 2
    fi
done

if [[ "${HACKATHON_ACCOUNT}" != "training2635" ]]; then
    echo "Expected HACKATHON_ACCOUNT=training2635; found ${HACKATHON_ACCOUNT}" >&2
    exit 2
fi

if [[ "${HACKATHON_PARTITION}" != "dc-gpu" ]]; then
    echo "Expected HACKATHON_PARTITION=dc-gpu; found ${HACKATHON_PARTITION}" >&2
    exit 2
fi

case "${HACKATHON_RESERVATION}" in
    challenge_3_and_5_day1|challenge_3_and_5_day2|challenge_3_and_5_day3)
        ;;
    *)
        echo "Unexpected Challenge 3 reservation: ${HACKATHON_RESERVATION}" >&2
        exit 2
        ;;
esac

if ! command -v scontrol >/dev/null 2>&1; then
    echo "scontrol is unavailable; run this preflight on a JURECA login node." >&2
    exit 2
fi

scontrol show reservation "${HACKATHON_RESERVATION}"

if [[ ! -x "${HACKATHON_ENV_DIR}/bin/python" ]]; then
    echo "Shared environment is missing or not executable: ${HACKATHON_ENV_DIR}" >&2
    exit 2
fi

if [[ ! -r "${HACKATHON_SHARED_ROOT}/data/SHA256SUMS" ]]; then
    echo "Shared data is missing or unreadable: ${HACKATHON_SHARED_ROOT}/data" >&2
    exit 2
fi

mkdir -p "${HACKATHON_RUN_ROOT}/runs" "${HACKATHON_RUN_ROOT}/slurm"

if [[ ! -e data && ! -L data ]]; then
    ln -s "${HACKATHON_SHARED_ROOT}/data" data
fi

if [[ ! -e runs && ! -L runs ]]; then
    ln -s "${HACKATHON_RUN_ROOT}/runs" runs
fi

(cd data && sha256sum -c SHA256SUMS)
python -m unittest discover -s tests -v

test_file="${HACKATHON_RUN_ROOT}/.preflight-write-${USER}-$$"
touch "${test_file}"
rm -f "${test_file}"

echo "Login-node preflight passed."
echo "Shared data: ${HACKATHON_SHARED_ROOT}/data"
echo "Run root: ${HACKATHON_RUN_ROOT}"
echo "Reservation: ${HACKATHON_RESERVATION}"
