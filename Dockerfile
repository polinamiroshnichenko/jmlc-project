FROM registry.example.com:443/dvps/ci-dungeon/python:3.14 as builder
RUN pip install uv
WORKDIR /app

COPY pyproject.toml pyproject.toml
RUN uv pip install --system .
COPY ./deployment/settings /etc/agents-workflow
COPY ./src /app/src
COPY ./assets /app/assets
COPY ./prompts /app/prompts


FROM builder as api-service
COPY run_api.py /app/run_api.py
CMD python run_api.py

FROM builder as worker-service
COPY run_worker.py /app/run_worker.py
CMD python run_worker.py
