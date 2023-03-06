FROM pkharchenkolab/numbat-rbase:latest
SHELL ["/bin/bash", "-c"]

RUN sudo apt-get install apt-transport-https ca-certificates gnupg curl -y

RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | tee /usr/share/keyrings/cloud.google.gpg && apt-get update -y && apt-get install google-cloud-sdk -y

ADD run_pileup_and_phase.R ~/