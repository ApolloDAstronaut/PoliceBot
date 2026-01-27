CUR_DIR = $(shell pwd)

build:
	docker build -t "$$(basename $(CUR_DIR))-duckiebot-image" --progress=plain .
	docker run --rm -v ./:/workspace --entrypoint=colcon "$$(basename $(CUR_DIR))-duckiebot-image" build

run:
	docker run --rm -it --network=host --privileged -v /dev/shm:/dev/shm -v ./:/workspace "$$(basename $(CUR_DIR))-duckiebot-image"

.PHONY: run build
