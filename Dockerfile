### Dockerfile for stainwarpy CLI tool

FROM python:3.11-slim

# set working directory
WORKDIR /app

# install stainwarpy
RUN pip install --no-cache-dir stainwarpy

# set entrypoint 
ENTRYPOINT [ "stainwarpy" ]

# default command
CMD [ "--help" ]
