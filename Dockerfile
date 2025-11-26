### Dockerfile for stainwarpy CLI tool

FROM python:3.11-slim

# set working directory
WORKDIR /app

# set stainwarpy version argument
ARG STAINWARPY_VERSION

# install stainwarpy
RUN pip install --no-cache-dir stainwarpy==${STAINWARPY_VERSION}

# allow both CLI usage and interactive testing
SHELL ["/bin/bash", "-c"]

# set entrypoint 
ENTRYPOINT [ "stainwarpy" ]

# default command
CMD [ "--help" ]
