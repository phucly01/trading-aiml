#!/bin/bash

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

cd ${SCRIPT_DIR}

echo "Creating Virtual Environment"

# if [ -z "`pip list | grep virtualenv`" ]; then
#     #Install virtualenv instead of using python -m env.  Virtualenv handles vboxsf better.
#     echo "Installing virtualenv module"
#     pip install virtualenv
# fi

envdirpath=${SCRIPT_DIR}/venv

python3 -m virtualenv ${envdirpath}
# virtualenv ./venv   #Try default options via symbolic links first
if [ "$?" -ne 0 ]; then
    echo "Symbolic links isn't allowed.  Trying copies"
    # virtualenv --always-copy ./venv
    python3 -m virtualenv --copies ${envdirpath}
    if [ "$?" -ne 0 ]; then
        rm -rf ${envdirpath}
        echo "--copies option isn't working.  Trying a different path"
        projname=`basename ${SCRIPT_DIR}`
        envdirpath="${HOME}/.${projname}venv"
        python3 -m virtualenv ${envdirpath}
    fi
fi

if [ -d ${envdirpath}/Scripts ]; then
    echo "Activating Virtual Environment"
    pushd ${envdirpath}/Scripts
    source ./activate
    popd
elif [ -d ${envdirpath}/bin ]; then
    echo "Activating Virtual Environment"
    pushd ${envdirpath}/bin
    source ./activate
    popd
fi

#requirements.txt can be created via pipreqs
#Update the version packages in requirements.txt via pip install -U -r requirements.txt
if [ -f ${SCRIPT_DIR}/requirements.txt ]; then
        echo "Installing required packages from requirements.txt"
        pip install -r ${SCRIPT_DIR}/requirements.txt
fi 