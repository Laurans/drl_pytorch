# drl_pytorch

## Requirements

* docker
* nvidia-docker
* pipenv


### To build Docker image

```
docker build - < Dockerfile
```


### Python libraries on the host machine

Python libraries are listed in `pipfile`. You can install it by using `pipenv install`.



## Usage

### Visdom

On the host machine, in another terminal session, you need to run `python -m visdom.server` if you want to visualize the training

### Docker 

```
docker run --runtime=nvidia -it --rm -v $PWD:/workdir -w /workdir --network=host --user="$(id -u):$(id -g)" drlnd_image python3 ./main.py
```