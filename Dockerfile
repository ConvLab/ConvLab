FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-devel 

# Install base packages.
RUN apt-get clean && apt-get update && apt-get install -y locales

RUN apt-get update --fix-missing && apt-get install -y \
    xvfb \
    bzip2 \
    ca-certificates \
    curl \
    gcc \
    git \
    libc-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    wget \
    libevent-dev \
    build-essential \
    openjdk-8-jdk && \
rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN [ "python", "-c", "import nltk; nltk.download('stopwords')" ]

WORKDIR /root

CMD ["/bin/bash"]
