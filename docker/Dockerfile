# Compute environment for the prompt-tuning reproduction.
# Base image ships torch 2.5.1 + CUDA 12.1 (runs on driver >= 525, e.g. 535/CUDA 12.2).
# Only dependencies are baked in; the repo code + data are bind-mounted at run time,
# so editing code needs no rebuild.
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# Avoid interactive tzdata etc.
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1

RUN pip install -U pip && pip install \
      "transformers>=4.44" \
      "datasets>=2.20" \
      accelerate \
      sentencepiece \
      huggingface_hub \
      tqdm \
      pyyaml

WORKDIR /workspace
