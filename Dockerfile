FROM pytorch/pytorch:latest

RUN apt-get update && apt-get install -y \
	python-pyglet \
        python3-opengl \
        zlib1g-dev \
        libjpeg-dev \
        patchelf \
        cmake \
        swig \
        libboost-all-dev \
        libsdl2-dev \
        libosmesa6-dev \
        xvfb \
        ffmpeg \
        wget && apt-get clean

RUN pip install --upgrade pip cython gym box2d visdom tqdm ipdb monkeytype

#EXPOSE 8097
ENV LC_ALL="C.UTF-8"
ENV LANG="C.UTF-8"