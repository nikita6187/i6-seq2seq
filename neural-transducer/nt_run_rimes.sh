#!/bin/bash
echo "Running RIMES training"
source ../env/bin/activate
# Using greedy alignments
# TODO: set back
#python nt_rimes.py CPU:0 15 False 15 False . False True
python nt_rimes.py CPU:0 15 False 15 False /home/nikita/i6-seq2seq/neural-transducer/checkpoint/test_rimes_20 False True
#python nt_rimes.py CPU:0 15 False 15 False /home/nikita/i6-seq2seq/neural-transducer/checkpoint/rimes_full_e1_chkpt_3440 False False
