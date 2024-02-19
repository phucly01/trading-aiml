#!/bin/bash

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

cd ${SCRIPT_DIR}

python -m venv ./env

if [ -d ${SCRIPT_DIR}/env/Scripts ]; then
   pushd ${SCRIPT_DIR}/env/Scripts
   ./activate

   if [ -f ${SCRIPT_DIR}/requirements.txt ]; then
        pip install -r ${SCRIPT_DIR}/requirements.txt
fi