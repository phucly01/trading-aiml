Prerequisites:

- Install mongodb server.  I have tried mongodb on Ubuntu 20 and 22, but neither of them worked.  The mongod cored.  So I used mongo container instead.  Here is the procedure for installing Docker on Oracle Linux 9.
  - Install Oracle Linux 9 OS on a VM or hardware.
  - Set the network interface as bridge and select the physical interface.
  - Install Docker:
    - sudo dnf install -y dnf-utils zip unzip
    - sudo dnf config-manager --add-repo=https://download.docker.com/linux/centos/docker-ce.repo
    - sudo dnf remove -y runc
    - sudo dnf install -y docker-ce --nobest
    - Create an external drive and share it as folder in the VM.  I used Virtual Box:
      - Create a folder or virtual drive.  In the VM setting, create share folder and point to the virtual drive or folder.  Set the mount point name like /Containers  (You will see this directory in the VM). 
    - Docker uses /var/lib/docker to store containers so, create a linker: 
      - ln -s /Containers /var/lib/docker 
  - Create and configure mongo container:
    - 