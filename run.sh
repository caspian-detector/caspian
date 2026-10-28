#!/bin/bash
# CASPIAN experiment runner — suppresses AutoGen/ONNX shutdown noise
python -m experiments.run_matrix "$@" 2>/dev/null