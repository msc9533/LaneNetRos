docker run -it --rm \
	--network host \
	-p 11311:11311 \
	--gpus all \
	--privileged \
	-e DISPLAY=unix$DISPLAY \
	--env="QT_X11_NO_MITSHM=1" \
	--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
	--volume=$(pwd):/home/catkin_ws/src \
	--device=/dev/video0:/dev/video0 \
    --name="lanenet-container" \
    lanenet-ros:latest bash