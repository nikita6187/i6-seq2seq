#!/bin/bash
echo "Running RIMES training"
source ../env/bin/activate
# Using greedy alignments
# TODO: set back
#python nt_rimes.py CPU:0 15 False 15 False . False True
python nt_rimes.py CPU:0 15 False 15 False /home/nikita/i6-seq2seq/neural-transducer/checkpoint/rime_model_reuse_chkpt_60 False False
