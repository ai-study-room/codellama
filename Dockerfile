FROM slob/cann:sha-dda4787-py38

WORKDIR /codellama

RUN apt-get update -y && \
    apt-get install -y vim

RUN pip install --upgrade pip


COPY . /codellama/
RUN pip3 install --upgrade setuptools
RUN pip3 install -e /codellama/.


