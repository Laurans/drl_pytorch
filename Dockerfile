FROM pytorch/pytorch:latest

RUN apt-get update && apt-get install -y \
        swig && apt-get clean

ENV LC_ALL="C.UTF-8"
ENV LANG="C.UTF-8"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && pip install --upgrade pip