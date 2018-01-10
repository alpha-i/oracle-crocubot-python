#!/usr/bin/env bash

CONDA_ENV=$1

conda create -y -n ${CONDA_ENV} python=3.5 numpy=1.14.0

source activate ${CONDA_ENV}

pip install -r requirements.txt
pip install tensorflow-gpu==1.4.0

python setup.py develop

RUNTIME_DIR=`pwd`/runtime

cd ${RUNTIME_DIR}

LAST_META=`ls -1t *_train_crocubot.meta | head -1`

LAST_META_TRIMMED="$(echo -e "${LAST_META}" | tr -d '[:space:]')"

if [ -z "${LAST_META_TRIMMED}" ];
then
    echo "No train files. Checkpoint file not created"
else
    IFS='.' read -r -a FULL_FILE_NAME <<< ${LAST_META_TRIMMED}
    MODEL_FILE_NAME=${FULL_FILE_NAME[0]}


    CHECKPOINT_FILE="model_checkpoint_path: \"${RUNTIME_DIR}/${MODEL_FILE_NAME}\"\nall_model_checkpoint_paths: \"${RUNTIME_DIR}/${MODEL_FILE_NAME}\""
    echo -e ${CHECKPOINT_FILE} > ${RUNTIME_DIR}/checkpoint
fi

