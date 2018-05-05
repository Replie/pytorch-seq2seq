#!/usr/bin/env bash

APP="replie-pythorch"
EXEC_FILE="replie"
VENV_PATH="/home/project10/project/pytorch/bin/activate"
PROJECT_DIR="/home/project10/project/"
APP_DIR="/home/project10/project/"${APP}


TRAIN_PATH=${APP_DIR}'dataset/'
EXPERIMENT_DIR="/home/project10/project/experiment/"
#update Code
cd ${APP_DIR}
git pull
cd ..

# set env vars
source ${VENV_PATH}
export PYTHONPATH=${APP_DIR}

# run train (--debug)
python ${APP_DIR}/${EXEC_FILE} --train_path ${TRAIN_PATH} --expt_dir EXPERIMENT_DIR  ${1} &
disown %1