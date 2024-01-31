docker build -t minkowski_engine .
#docker run minkowski_engine python3 -c "import MinkowskiEngine; print(MinkowskiEngine.__version__)"
docker run -it --rm --gpus all --net host --shm-size=1g -v  $(pwd)/..:/code -v /data3:/data3 minkowski_engine
