#!/bin/bash
echo "Running RIMES training"
source ../env/bin/activate
# Using greedy alignments
# TODO: set back to cpu
python nt_rimes.py CPU:0 15 False 15 False . False True

