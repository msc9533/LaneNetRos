FROM tensorflow/tensorflow:1.15.0-gpu-py3

RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get update \
  && DEBIAN_FRONTEND=noninteractive apt-get -y install \
    wget \
    curl \
    python3-pip \
    git

RUN python3 -m pip install --upgrade pip
RUN pip install glog==0.3.1 \
pip install loguru==0.2.5 \
pip install tqdm==4.28.1 \
pip install matplotlib==2.2.4 \
pip install opencv_contrib_python==4.2.0.32 \
pip install numpy==1.16.4 \
pip install scikit_learn==0.24.1 \
pip install PyYAML==5.4.1

RUN /bin/sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV ROS_DISTRO=melodic
RUN apt-get update \
  && DEBIAN_FRONTEND=noninteractive apt-get -y install \
    ros-melodic-desktop-full
RUN apt-get update \
  && DEBIAN_FRONTEND=noninteractive apt-get -y install python-rosdep \
    python-rosinstall \
    python-rosinstall-generator \
    python-wstool \
    build-essential
RUN apt-get update \
  && DEBIAN_FRONTEND=noninteractive apt-get -y install python3-pip \
    python3-all-dev \
    python3-rospkg
RUN apt install -y ros-melodic-desktop-full --fix-missing

WORKDIR /
RUN wget https://gist.githubusercontent.com/msc9533/0ff4df052811bbb1a848309a69df74fe/raw/f0b47e4cfb86a15b63c58d59531a9747bde2751c/ros_entrypoint.sh
RUN chmod a+x ros_entrypoint.sh
WORKDIR /home/catkin_ws
RUN rm -rf /var/lib/apt/lists/*
ENV LD_LIBRARY_PATH = $LD_LIBRARY_PATH:/usr/local/cuda/lib64
ENV PATH=$PATH:/home/catkin_ws/lanenet-lane-detection
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]