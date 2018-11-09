# drl_pytorch

## Requirements

* docker
* nvidia-docker
* pipenv

### Python libraries on the host machine

Python libraries are listed in `pipfile`

## Usage

### Vidom

On the host machine, you need to run `python -m visdom.server`

### Docker 

```
docker run --runtime=nvidia -it --rm -v $PWD:/workdir -w /workdir --network=host drlnd_image python3 ./main.py
```