FROM nvidia/cuda:12.3.2-runtime-ubuntu22.04
WORKDIR /app
COPY . .
RUN apt update
RUN apt install -y git libsndfile1-dev tesseract-ocr espeak-ng python3.10 python3-pip ffmpeg wget
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
RUN dpkg -i cuda-keyring_1.1-1_all.deb
RUN apt-get -y install cuda-toolkit-12-4
RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir "jax==0.4.16" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip install --no-cache-dir torch torchvision torchaudio transformers pandas datasets sentence-transformers evaluate optuna imblearn lime
RUN pip install --no-cache-dir git+https://github.com/facebookresearch/detectron2.git pytesseract
RUN pip install pip install -U "itsdangerous<2.1.0"