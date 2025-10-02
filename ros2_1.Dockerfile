####################################################################################################
# Stage: deps
FROM artifactory.ar.int:5014/ros:humble-ros-base AS deps
# FROM ros:humble-ros-base AS deps

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    clang-14 \
    clang-format-14 \
    clang-tidy-14 \
    libeigen3-dev \
    libignition-gazebo6-dev \
    libpoco-dev \
    python3-pip \
    ros-humble-ament-clang-format \
    ros-humble-ament-cmake-clang-format \
    ros-humble-ament-cmake-clang-tidy \
    ros-humble-ament-flake8 \
    ros-humble-angles \
    ros-humble-moveit \
    ros-humble-pinocchio \
    ros-humble-control-msgs \
    ros-humble-control-toolbox \
    ros-humble-controller-interface \
    ros-humble-controller-manager \
    ros-humble-generate-parameter-library \
    ros-humble-hardware-interface \
    ros-humble-hardware-interface-testing \
    ros-humble-launch-testing \
    ros-humble-realtime-tools \
    ros-humble-ros2-control \
    ros-humble-ros2-control-test-assets \
    ros-humble-ros2-controllers \
    ros-humble-xacro \
    ros-humble-rosbag2-storage-mcap \
    ros-humble-foxglove-bridge \
    ros-humble-compressed-image-transport \
    ros-humble-xacro \
    ros-humble-joint-state-publisher \
    ros-humble-plotjuggler-ros \
    can-utils \
    iproute2 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install additional dependencies for robotpkg-pinocchio (required for libfranka)
RUN apt-get install -y lsb-release curl \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL http://robotpkg.openrobots.org/packages/debian/robotpkg.asc | tee /etc/apt/keyrings/robotpkg.asc \
    && echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/robotpkg.asc] http://robotpkg.openrobots.org/packages/debian/pub $(lsb_release -cs) robotpkg" | tee /etc/apt/sources.list.d/robotpkg.list \
    && apt-get update \
    && apt-get install -y robotpkg-pinocchio

# Set Python environment variable to flush logs
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends curl iputils-ping vim-tiny wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
WORKDIR /workspace
RUN mkdir -p /ws/src/
RUN source /opt/ros/humble/setup.bash && \
    echo "ROS env OK" && \
    ros2 --help >/dev/null



CMD ["/bin/bash"]
