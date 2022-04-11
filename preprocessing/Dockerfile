RUN pip install -U pyyaml
ARG DISABLE_CACHE
ADD . /app/

WORKDIR /app
ENTRYPOINT ["python", "write-yaml.py" ]


