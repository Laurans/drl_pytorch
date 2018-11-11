# drl_pytorch

## Requirements

* docker
* nvidia-docker

For python on the host machine, you can use pipenv with pyenv
* [pyenv installer](https://github.com/pyenv/pyenv-installer), [more about pyenv](https://github.com/pyenv/pyenv)
* [pipenv](https://pipenv.readthedocs.io/en/latest/)


### To build Docker image

```
docker build - < Dockerfile
```


### Python libraries on the host machine

Python libraries are listed in `Pipfile`. You can install it by using 
```
pipenv install
```



## Usage

### Visdom

On the host machine, in another terminal session, if you want to visualize the training, you need to run 
```
python -m visdom.server
``` 

### Docker 
```
docker run --runtime=nvidia -it --rm -v $PWD:/workdir -w /workdir --network=host --user="$(id -u):$(id -g)" drlnd_image python3 ./main.py
```