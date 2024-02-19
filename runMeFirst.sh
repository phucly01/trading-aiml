#!/bin/bash

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

cd ${SCRIPT_DIR}

echo "Creating Virtual Environment"

if [ -z "`pip list | grep virtualenv`" ]; then
    #Install virtualenv instead of using python -m env.  Virtualenv handles vboxsf better.
    echo "Installing virtualenv module"
    pip install virtualenv
fi

#python -m venv --copies ./env
virtualenv ./env   #Try default options via symbolic links first
if [ "$?" -ne 0 ]; then
    echo "Symbolic links isn't allowed.  Trying copies"
    virtualenv --always-copy ./env
fi

if [ -d ${SCRIPT_DIR}/env/Scripts ]; then
    echo "Activating Virtual Environment"
    pushd ${SCRIPT_DIR}/env/Scripts
    ./activate
    popd
elif [ -d ${SCRIPT_DIR}/env/bin ]; then
    echo "Activating Virtual Environment"
    pushd ${SCRIPT_DIR}/env/bin
    ./activate
    popd
fi

#requirements.txt can be created via pipreqs
#Update the version packages in requirements.txt via pip install -U -r requirements.txt
if [ -f ${SCRIPT_DIR}/requirements.txt ]; then
        echo "Installing required packages from requirements.txt"
        pip install -r ${SCRIPT_DIR}/requirements.txt
fi 