FROM pytorch/pytorch:latest

ENV LC_ALL="C.UTF-8"
ENV LANG="C.UTF-8"

RUN apt-get update && apt-get install -y \
        python-pyglet \
        python3-opengl \
        x11vnc \
        swig \
        xvfb \
        wget \ 
        fluxbox\
        wmctrl\
        git && apt-get clean

COPY bootstrap.sh /
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt && pip install --upgrade pip && chmod +x "/bootstrap.sh" && mkdir /tmp/.X11-unix && chmod 1777 /tmp/.X11-unix

ENTRYPOINT ["/bootstrap.sh"]