FROM continuumio/miniconda3:22.11.1

# Make RUN commands use `bash --login`:
SHELL ["/bin/bash", "--login", "-c"]

RUN pip install -U adpbulk gseapy rich statannotations scanpy fastcluster torch
RUN pip install --upgrade --index-url https://test.pypi.org/simple/ scbtf

# install gcloud
RUN apt-get update && apt-get install --no-install-recommends -y curl alien unzip build-essential dpkg-dev gnupg lsb-release procps && \
    export CLOUD_SDK_REPO="cloud-sdk-$(lsb_release -c -s)" && \
    echo "deb http://packages.cloud.google.com/apt $CLOUD_SDK_REPO main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    apt-get update && apt-get install -y google-cloud-sdk