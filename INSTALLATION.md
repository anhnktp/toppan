# Project5_ExtractReID

 ExactReID is a project bult on Python3.6 for a Deeplearning Tracking technology
 It features: 
  - Person Detection for Camera at Shelf
  - Person Detection for 360-degree camera
  - Pose Extraction for Camera at Shelf
  - Camera Tracking for 360-degree camera
  - Event Notification
  
# Setup Docker if Docker is not installed

## I. Prerequisites

### OS requirement

- Bionic 16.04 LTS (Highly recommended)

### Hardware requirement

- GPU device

### Software requirements

- NVIDIA Graphics Driver 430
- Docker, docker-compose and nvdia-docker2

## II. Setup Instructions for Host Machine

### 1. Install NVIDIA Driver

If `/etc/modprobe.d/disable-nouveau.conf` doesn't exist or something like that, you can create it and add these lines into the file manually.

```
blacklist nouveau
options nouveau modeset=0
```

```bash
echo $'blacklist nouveau\noptions nouveau modeset=0' >> /etc/modprobe.d/disable-nouveau.conf

sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
sudo apt install -y nvidia-driver-430

```

### 2. Install docker

#### Uninstall old versions

Older versions of Docker were called `docker`, `docker.io` , or `docker-engine`. If these are installed, uninstall them:

```bash
sudo apt-get remove docker docker-engine docker.io containerd runc
```

#### Install Docker CE

```bash

curl -sSL get.docker.io | bash -

sudo usermod -aG docker $USER
```

### 3. Install docker-compose

```bash
sudo pip3 install docker-compose
```

### 4. NVIDIA Container Runtime for Docker (nvdia-docker2)

Make sure you have installed the [NVIDIA driver](https://github.com/NVIDIA/nvidia-docker/wiki/Frequently-Asked-Questions#how-do-i-install-the-nvidia-driver) and a [supported version](https://github.com/NVIDIA/nvidia-docker/wiki/Frequently-Asked-Questions#which-docker-packages-are-supported) of [Docker](https://docs.docker.com/engine/installation/) for your distribution (see [prerequisites](https://github.com/NVIDIA/nvidia-docker/wiki/Installation-(version-2.0)#prerequisites)).

If you have a custom `/etc/docker/daemon.json`, the `nvidia-docker2` package might override it.

```bash
# If you have nvidia-docker 1.0 installed: we need to remove it and all existing GPU containers
docker volume ls -q -f driver=nvidia-docker | xargs -r -I{} -n1 docker ps -q -a -f volume={} | xargs -r docker rm -f
sudo apt-get purge -y nvidia-docker

# Add the package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update

# Install nvidia-docker2 and reload the Docker daemon configuration
sudo apt-get install -y nvidia-docker2
sudo pkill -SIGHUP dockerd

# Test nvidia-smi with the latest official CUDA image
docker run --runtime=nvidia --rm nvidia/cuda:9.0-base nvidia-smi
```

### 5. Reboot your host machine to accept all changes above

```bash
sudo reboot
```

### 6. Setup host machine for build and running our docker container
- Setup to show GUI tracking 360 cam (X11 forwarding mode)

    - Modify [.env](.env) file
        - SHOW_GUI=True

    - Setup X11

~~~bash
$ XSOCK=/tmp/.X11-unix
$ XAUTH=/tmp/.docker.xauth
$ touch ${XAUTH}
$ xauth nlist ${DISPLAY} | sed 's/^..../ffff/' | xauth -f ${XAUTH} nmerge -
~~~

- Build and start the docker container

~~~bash
# Adds docker to X server access control list (ACL)
$ xhost + local:docker

# Build and run the services
$ docker-compose up --build

# Removes docker out of X server ACL if you are not working with
$ xhost - local:docker
 ~~~