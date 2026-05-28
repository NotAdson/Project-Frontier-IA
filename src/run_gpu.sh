#!/bin/bash

# Dynamically find all NVIDIA pip library paths in the active Python environment
NVIDIA_PATHS=$(python3 -c "
import os, sys
try:
    import nvidia
    nv_dir = os.path.dirname(nvidia.__file__)
    paths = [os.path.join(nv_dir, d, 'lib') for d in os.listdir(nv_dir) 
             if os.path.isdir(os.path.join(nv_dir, d, 'lib'))]
    print(':'.join(paths))
except ImportError:
    print('')
")

# Export them to the dynamic linker path
export LD_LIBRARY_PATH="$NVIDIA_PATHS:$LD_LIBRARY_PATH"

# Run the passed python module or script
python3 "$@"
