### Dockerfile for stainwarpy CLI tool

ARG STAINWARPY_VERSION=latest

FROM python:3.11-slim

# set working directory
WORKDIR /app

# install stainwarpy
RUN pip install --no-cache-dir stainwarpy==${STAINWARPY_VERSION}

# set entrypoint 
ENTRYPOINT [ "stainwarpy" ]

# default command
CMD [ "--help" ]
