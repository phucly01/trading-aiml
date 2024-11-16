#About
This project utilizes some of the machine learning models from TensorFlow for stock prediction.  


#Project Preparation
##Prerequisites

- Create virtual environment:
  - Note: This project was created via Python 3
  - For Windows:
    - At the project directory run 'python -m venv venv'
    - Run venv/Scripts/activate.bat
  
- Install TensorFlow:  TensorFlow can be installed via pip install tensorflow or python -m pip install sensorflow.
  - Note: Make sure the python version is as expected by TensorFlow or there is a high chance it won't work.
  - For Windows:
    - Run 'pip install tensorflow'
  - Clone tensorflow
    - Clone from git clone https://github.com/tensorflow/tensorflow.git
    - If a specific version of tensorflow is desired, use 'git checkout <branch>' (run git branch -r to see list of branch names)
    - Go to the tensorflow directory and run 'python configure.py'
      - During the configuration there are a couple of options to pick from:
        - ROCm - this provides a set of supports for utilizing GPU.  You may want to say no to this if you are using VM or on old computers where GPU isn't available.
       
        - 
- Install cassandra database via docker container.
  - Install Oracle Linux 9 OS on a VM or hardware.
  - Set the network interface as bridge and select the physical interface.
  - Install Docker:
    - sudo dnf install -y dnf-utils zip unzip
    - sudo dnf config-manager --add-repo=https://download.docker.com/linux/centos/docker-ce.repo
    - sudo dnf remove -y runc
    - sudo dnf install -y docker-ce --nobest
  - Create and configure cassandra container:


- Build via pyinstaller:
  - pyinstaller -p data -p aiml -p analysis -p chart -p config -p data -p db -p models main.py
  - add --onefile to generate a single executable (large size executables)
