#!/usr/bin/env bash
export PYTHONPATH=${PYTHONPATH}

log_folder=logs
mkdir -p ${log_folder}


for exp in 1 2 3 4 5
do
  for frac in '0.1'
  do
      identifier='nonIID_mlp_cifar_C'${frac}'_E10_B10'
      python ../src/federated_main.py --conf_file_name=${identifier}'.yaml' \
                              | tee ${log_folder}/${identifier}'_exp'${exp}.txt
  done
done


