#!/bin/bash
echo "Running RIMES training"
source ../env/bin/activate
# Using greedy alignments
python nt_rimes.py CPU:0 15 False 15 False . False True

