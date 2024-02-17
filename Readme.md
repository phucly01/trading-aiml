Prerequisites:

- Install cassandra database via docker container.
  - Install Oracle Linux 9 OS on a VM or hardware.
  - Set the network interface as bridge and select the physical interface.
  - Install Docker:
    - sudo dnf install -y dnf-utils zip unzip
    - sudo dnf config-manager --add-repo=https://download.docker.com/linux/centos/docker-ce.repo
    - sudo dnf remove -y runc
    - sudo dnf install -y docker-ce --nobest
  - Create and configure cassandra container:
    - 