FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

# Eviter prompts
ENV DEBIAN_FRONTEND=noninteractive

# Outils système + librairies X (Pillow/OpenCV), ffmpeg, build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libsm6 libxext6 libgl1 build-essential \
 && rm -rf /var/lib/apt/lists/*

# Cloner StyleGAN3 (legacy, dnnlib, torch_utils)
RUN git clone --depth 1 https://github.com/NVlabs/stylegan3 /opt/stylegan3
ENV PYTHONPATH=/opt/stylegan3:$PYTHONPATH

# Dépendances Python
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Pré-compilation des extensions StyleGAN3 (JIT) pour CUDA (L4 = sm_89)
ENV TORCH_CUDA_ARCH_LIST="8.9"
ENV FORCE_CUDA=1
RUN python - <<'PY'
import sys
sys.path.append("/opt/stylegan3")
# importer les ops pour forcer la JIT-compile
from torch_utils.ops import bias_act, filtered_lrelu
print(">> StyleGAN3 custom ops import OK")
PY

# Appli
COPY app.py /app/app.py

# Dossier modèles monté par volume
RUN mkdir -p /models

# Port Gradio
EXPOSE 7862
ENV PORT=7862

# Variables par défaut (surchargées par docker-compose/env)
ENV NETWORK=/models/network-snapshot.pkl
ENV ENABLE_GFPGAN=1
ENV GFPGAN_MODEL=/models/GFPGANv1.4.pth

# Lancement
CMD ["python", "app.py"]
