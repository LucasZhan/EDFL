#!/usr/bin/env bash
export PYTHONPATH=${PYTHONPATH}

log_folder=logs
mkdir -p ${log_folder}


for exp in {1..50}
do
  for frac in '0.1'
  do
      identifier='nonIID_cnn_cifar_C'${frac}'_E10_B10_CPU0.1'
      python ../src/federated_main.py --conf_file_name=${identifier}'.yaml' \
                              | tee ${log_folder}/${identifier}'_exp'${exp}.txt
  done
done

for exp in {1..50}
do
  for frac in '0.1'
  do
      identifier='nonIID_cnn_mnist_C'${frac}'_E10_B10_CPU0.1'
      python ../src/federated_main.py --conf_file_name=${identifier}'.yaml' \
                              | tee ${log_folder}/${identifier}'_exp'${exp}.txt
  done
done
