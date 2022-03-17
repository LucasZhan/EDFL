#!/usr/bin/env bash
export PYTHONPATH=${PYTHONPATH}

log_folder=logs
mkdir -p ${log_folder}


for exp in 1
do
  for frac in '0.2' '0.4' '0.6' '0.8' '1.0'
  do
      identifier='nonIID_cnn_mnist_C'${frac}'_E10_B10'
      python src/federated_main.py --conf_file_name=${identifier}'.yaml' \
                              | tee ${log_folder}/${identifier}'_exp'${exp}.txt
  done
done


