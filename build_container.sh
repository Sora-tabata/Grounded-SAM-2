#!/bin/bash

# UI permissions setup
XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth
touch $XAUTH
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -
xhost +local:docker

# Stop and remove the existing container if it exists
if [ $(docker ps -a -q -f name=gsam2) ]; then
    docker stop gsam2
    docker rm gsam2
fi

# Create a new container with GPU support, GUI support, and device mounting for webcam access
docker run -td --privileged --net host --ipc host \
    --name="gsam2" \
    --gpus all \
    --shm-size 12gb \
    --device=/dev/video0:/dev/video0 \
    -e "DISPLAY=$DISPLAY" \
    -e "QT_X11_NO_MITSHM=1" \
    -v "$XSOCK:$XSOCK" \
    -e "XAUTHORITY=$XAUTH" \
    --cap-add=SYS_PTRACE \
    -v /etc/group:/etc/group:ro \
    -v /dev/bus/usb:/dev/bus/usb \
    -v $HOME:/mnt/source \
    -v /media/sora:/mnt/media \
    sorajang/gsam2 bash

# Wait for the container to start
sleep 5

# Install necessary packages and setup the environment inside the container
docker exec -it gsam2 bash -i -c "
set -e  # Exit on error
apt-get -y update && apt-get -y upgrade && \
apt-get install -y libgl1-mesa-dev libglib2.0-0 v4l-utils && \
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia && \
pip3 install git+https://github.com/cocodataset/panopticapi.git && \
pip3 install git+https://github.com/mcordts/cityscapesScripts.git && \
pip3 install mxnet-mkl==1.6.0 numpy==1.23.1 && \
cd /mnt/source/gsam2 && \
pip install -e . && \
pip install --no-build-isolation -e grounding_dino && \
python setup.py build_ext --inplace
"

# Reset UI permissions after container setup
xhost -local:docker

echo "Docker container 'gsam2' setup is complete."