#!/bin/sh

export PYTHONPATH=./
PYTHON=python3

TRAIN_CODE=train.py

dataset=s3dis
exp_name=$1
config_name=$2

exp_dir=exp/${dataset}/${exp_name}
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result
config=${exp_dir}/${config_name}.yaml

mkdir -p ${model_dir} ${result_dir}
mkdir -p ${result_dir}/last
mkdir -p ${result_dir}/best
cp config/${dataset}/res.yaml tool/train.sh tool/${TRAIN_CODE} tool/test.sh ${exp_dir}


now=$(date +"%Y%m%d_%H%M%S")
$PYTHON ${exp_dir}/${TRAIN_CODE} \
  --config=${config} \
  save_path ${exp_dir} \
  2>&1 | tee ${exp_dir}/train-$now.log
