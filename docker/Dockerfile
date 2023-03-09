FROM continuumio/miniconda3:22.11.1

# Make RUN commands use `bash --login`:
SHELL ["/bin/bash", "--login", "-c"]

# Create the environment:
COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml

# Initialize conda in bash config fiiles:
RUN conda init bash && source ~/.bashrc
RUN conda config --set auto_activate_base false
RUN echo 'conda activate minim_pyro' >> ~/.bashrc

# Install scbtf
RUN pip install --index-url https://test.pypi.org/simple/ scbtf==0.1.2 --user

# install gcloud
RUN apt-get update && apt-get install --no-install-recommends -y curl alien unzip build-essential dpkg-dev gnupg lsb-release procps && \
    export CLOUD_SDK_REPO="cloud-sdk-$(lsb_release -c -s)" && \
    echo "deb http://packages.cloud.google.com/apt $CLOUD_SDK_REPO main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    apt-get update && apt-get install -y google-cloud-sdk