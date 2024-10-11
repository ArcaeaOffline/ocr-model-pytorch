FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

COPY requirements.txt /tmp/requirements.txt

RUN apt update \
    && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC \
        apt install -y tzdata build-essential \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r /tmp/requirements.txt -i https://mirrors.bfsu.edu.cn/pypi/web/simple
