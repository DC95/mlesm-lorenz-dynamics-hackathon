#!/bin/bash

# This file is sourced by setup.sh and activate.sh. Keep the module list small
# and pin versions that have been rehearsed together on JURECA-DC.
if ! command -v module >/dev/null 2>&1; then
    echo "The JSC environment-modules command is unavailable." >&2
    return 2 2>/dev/null || exit 2
fi

module purge &&
    module load Stages/2026 &&
    module load GCCcore/14.3.0 &&
    module load PyTorch/2.9.1 &&
    module load matplotlib/3.10.5
