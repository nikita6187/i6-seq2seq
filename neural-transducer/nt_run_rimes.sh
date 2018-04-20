#!/bin/bash
echo "Running RIMES training"
source ../env/bin/activate
# Using greedy alignments
# TODO: set back
#python nt_rimes.py CPU:0 15 False 15 False . False True
python nt_rimes.py CPU:0 15 False 15 False ~/i6-seq2seq/neural-transducer/checkpoint/2nd_full_run/rimes_2_full_rough_18360 False True

# Checklist for deployment:
# - (Greedy) alignments
# - Correct amount of CPUs
# - Full RIMES dataset
# - Load in correct model
# - send email
# - work in current directory
# - time
# - h_vmem
# - num_proc
# - using latest version
# - saving checkpoint to correct location
# - saving log to correct location
# - correct NT model
# - batch size correct
# - qsub output file correctly set
# - check if there is enough memory
#qsub -cwd -l h_vmem=40G -l h_rt=150:00:00 -l num_proc=16 -m abe -o ./rimes/rimes_2

