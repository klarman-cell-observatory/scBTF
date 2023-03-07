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
RUN pip install --index-url https://test.pypi.org/simple/ scbtf==0.1.1 --user
