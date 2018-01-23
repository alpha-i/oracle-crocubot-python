#!/usr/bin/env bash

CONDA_ENV=$1

conda create -n $CONDA_ENV python=3.5 numpy=1.14.0

source activate $CONDA_ENV

pip install -r requirements.txt
pip install tensorflow-gpu

python setup.py develop


RUNTIME_DIR=`pwd`/runtime

LST_META=`ls -1t ${RUNTIME_DIR}/*_train_crocubot.meta | head -1`
IFS='.' read -r -a FULL_FILE_NAME <<< ${LAST_META}
MODEL_FILE_NAME=${FULL_FILE_NAME[0]}

CHECKPOINT_FILE="model_checkpoint_path: \"${RUNTIME_DIR}/${MODEL_FILE_NAME}\"\nall_model_checkpoint_paths: \"${RUNTIME_DIR}/${MODEL_FILE_NAME}\""
echo -e ${CHECKPOINT_FILE} > ${RUNTIME_DIR}/checkpoint


