#!/bin/sh

# Source this script with 'source set_pytorch_path'

export LIBTORCH=$(python3 -c 'import torch; from pathlib import Path; print(Path(torch.__file__).parent)')
export DYLD_FALLBACK_LIBRARY_PATH="${LIBTORCH}/lib"
