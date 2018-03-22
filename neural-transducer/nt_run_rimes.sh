#!/bin/bash
echo "Running RIMES training"
source ../env/bin/activate
# Using greedy alignments
python nt_rimes.py CPU:0 8 False 8 False . False True

