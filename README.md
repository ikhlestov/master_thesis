to start docker on server

```bash
docker run -it -p 9595:8888 --gpus all -v `pwd`:/src tensorflow/tensorflow:latest-gpu
```

within container
```bash
pip install jupyter
jupyter notebook --allow-root --ip=0.0.0.0
```

follow to 
http://213.32.26.74:9595/?token=
