<<<<<<< HEAD
FROM pandoc/latex:latest-ubuntu

# packages for devcontainer
RUN apt-get update 
RUN apt-get install -y python3 python3-pip git curl

WORKDIR /tmp

# python dependencies for build.py
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# install tectonic
RUN curl --proto '=https' --tlsv1.2 -fsSL https://drop-sh.fullyjustified.net | sh
RUN mv tectonic /usr/bin/
RUN wget http://nz2.archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2.19_amd64.deb
RUN dpkg -i libssl1.1_1.1.1f-1ubuntu2.19_amd64.deb
=======
FROM pandoc/latex:latest

ARG USERNAME=markdown-warrior
ARG USER_UID=1000
ARG USER_GID=${USER_UID}

ARG INSTALL_ZSH="true"

WORKDIR /tmp

# packages for devcontainer
RUN apk update
RUN apk add python3 py3-pip git curl

# install devcontainer utils
RUN curl https://raw.githubusercontent.com/microsoft/vscode-dev-containers/main/script-library/common-alpine.sh > common-alpine.sh
RUN ash common-alpine.sh "${INSTALL_ZSH}" "${USERNAME}" "${USER_UID}" "${USER_GID}" \
    && rm common-alpine.sh

# install mermaid dependencies
RUN apk add npm
RUN npm install --global mermaid-filter

# install and test tectonic
RUN curl --proto '=https' --tlsv1.2 -fsSL https://drop-sh.fullyjustified.net | sh
RUN mv tectonic /usr/bin/
RUN tectonic --version

USER $USERNAME

# install academic_markdown for the user
RUN pip3 install academic_markdown
>>>>>>> 36e34ce5d688ad0f6031d133421cac709756d06c
