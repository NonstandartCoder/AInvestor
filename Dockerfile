FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

ENV TZ=Europe/Moscow \
    DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/home/project/.local/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
        curl ca-certificates git build-essential \
        libxml2 libxslt1.1 tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && useradd -m -s /bin/bash project \
    && curl -LsSf https://astral.sh/uv/install.sh \
        | su - project -s /bin/bash -c "env UV_INSTALL_DIR=/home/project/.local/bin sh"

USER project
WORKDIR /srv/project

COPY --chown=project:project pyproject.toml uv.lock ./
RUN uv sync --locked --compile-bytecode --group eda --group dev

COPY --chown=project:project app ./app

CMD ["bash"]
