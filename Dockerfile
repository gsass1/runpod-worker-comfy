# Multi-stage image: default `docker build` / RunPod "build from Git" uses the last stage (`final`).
# Downloader stage ARG MODEL_TYPE (below) must have a default or RunPod/GitHub builds ship ComfyUI with no checkpoints.
#
# Character-sheet custom nodes (MVAdapter + Impact Pack) default to ON so a plain `docker build` or RunPod
# "build from Git" without extra args matches the bundled character-sheet workflow (LdmPipelineLoader, FaceDetailer, …).
# For a slimmer image: --build-arg WITH_CHARACTER_SHEET_NODES=false
#
# Default MODEL_TYPE (downloader) is character-sheet so animagine-xl-3.1, 4x-UltraSharp, face_yolov8m ship in a plain build.
# Override with --build-arg MODEL_TYPE=sdxl (or flux / sd3) if you need a different stack.
# Optional for sd3 / flux1-dev: HUGGINGFACE_ACCESS_TOKEN

# Stage 1: Base image with common dependencies
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as base

# When true, installs ComfyUI-MVAdapter + Impact Pack (character-sheet workflows).
ARG WITH_CHARACTER_SHEET_NODES=true
# Pin ComfyUI-MVAdapter to a release tag by default (reproducible; main can break ComfyUI 0.2.7).
# Override with --build-arg COMFYUI_MV_ADAPTER_REF=main to track upstream.
ARG COMFYUI_MV_ADAPTER_REF=v1.0.2
# Impact Pack Main requires newer ComfyUI (SCHEDULER_HANDLERS); 8.9 matches ComfyUI 0.2.7.
# Override with --build-arg COMFYUI_IMPACT_PACK_REF=Main after upgrading comfy-cli --version.
ARG COMFYUI_IMPACT_PACK_REF=8.9

# Prevents prompts from packages asking for user input during installation
ENV DEBIAN_FRONTEND=noninteractive
# Prefer binary wheels over source distributions for faster pip installations
ENV PIP_PREFER_BINARY=1
# Ensures output from python is printed immediately to the terminal without buffering
ENV PYTHONUNBUFFERED=1 
# Speed up some cmake builds
ENV CMAKE_BUILD_PARALLEL_LEVEL=8

# Install Python, git and other necessary tools
# libglib2.0-0: provides libgthread-2.0.so.0 required by opencv (cv2) for Impact Pack / Subpack import
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# Clean up to reduce image size
RUN apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install comfy-cli
RUN pip install comfy-cli

# Install ComfyUI
RUN /usr/bin/yes | comfy --workspace /comfyui install --cuda-version 11.8 --nvidia --version 0.2.7

# Change working directory to ComfyUI
WORKDIR /comfyui

COPY src/install_character_sheet_custom_nodes.sh /tmp/install_character_sheet_custom_nodes.sh
RUN chmod +x /tmp/install_character_sheet_custom_nodes.sh && \
    if [ "$WITH_CHARACTER_SHEET_NODES" = "true" ]; then \
      COMFYUI_MV_ADAPTER_REF="$COMFYUI_MV_ADAPTER_REF" \
      COMFYUI_IMPACT_PACK_REF="$COMFYUI_IMPACT_PACK_REF" \
      /tmp/install_character_sheet_custom_nodes.sh; \
    fi

# Install Runpod serverless worker SDK.
RUN pip install runpod==1.9.1 requests

# Support for the network volume
ADD src/extra_model_paths.yaml ./

# Go back to the root
WORKDIR /

# Add scripts
COPY src/start.sh src/diagnose_custom_nodes_env.sh src/restore_snapshot.sh src/rp_handler.py test_input.json ./
RUN chmod +x /start.sh /diagnose_custom_nodes_env.sh /restore_snapshot.sh

# Optional ComfyUI Manager snapshot (see snapshots/README.md)
COPY snapshots/ /snapshots-build/
# POSIX /bin/sh: avoid relying on ls + glob exit status; copy first matching JSON if any
RUN for f in /snapshots-build/*snapshot*.json; do \
      if [ -f "$f" ]; then cp "$f" / && break; fi; \
    done

# Restore the snapshot to install custom nodes (no-op if no snapshot file in /)
RUN /restore_snapshot.sh

# Start container
CMD ["/start.sh"]

# Stage 2: Download models
FROM base as downloader

ARG HUGGINGFACE_ACCESS_TOKEN
# Default: character-sheet weights (see README). Override e.g. MODEL_TYPE=sdxl for stock SDXL-only checkpoints.
ARG MODEL_TYPE=character-sheet

# Change working directory to ComfyUI
WORKDIR /comfyui

# Create necessary directories (unet/clip required for flux targets; rest for sdxl / character-sheet)
RUN mkdir -p models/checkpoints models/vae models/unet models/clip \
    models/upscale_models models/ultralytics/bbox

# character-sheet must be built with WITH_CHARACTER_SHEET_NODES=true on base (same docker build)
RUN if [ "$MODEL_TYPE" = "character-sheet" ] && [ ! -d /comfyui/custom_nodes/ComfyUI-MVAdapter ]; then \
      echo "ERROR: MODEL_TYPE=character-sheet requires --build-arg WITH_CHARACTER_SHEET_NODES=true (ComfyUI-MVAdapter missing)." >&2; \
      exit 1; \
    fi

# Download checkpoints/vae/LoRA to include in image based on model type
RUN if [ "$MODEL_TYPE" = "sdxl" ]; then \
      wget -O models/checkpoints/sd_xl_base_1.0.safetensors https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors && \
      wget -O models/vae/sdxl_vae.safetensors https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors && \
      wget -O models/vae/sdxl-vae-fp16-fix.safetensors https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl_vae.safetensors; \
    elif [ "$MODEL_TYPE" = "character-sheet" ]; then \
      wget -O models/checkpoints/animagine-xl-3.1.safetensors https://huggingface.co/cagliostrolab/animagine-xl-3.1/resolve/main/animagine-xl-3.1.safetensors && \
      wget -O models/vae/sdxl-vae-fp16-fix.safetensors https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl_vae.safetensors && \
      wget -O models/upscale_models/4x-UltraSharp.pth https://huggingface.co/Kim2091/UltraSharp/resolve/main/4x-UltraSharp.pth && \
      wget -O models/ultralytics/bbox/face_yolov8m.pt https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8m.pt; \
    elif [ "$MODEL_TYPE" = "sd3" ]; then \
      wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/checkpoints/sd3_medium_incl_clips_t5xxlfp8.safetensors https://huggingface.co/stabilityai/stable-diffusion-3-medium/resolve/main/sd3_medium_incl_clips_t5xxlfp8.safetensors; \
    elif [ "$MODEL_TYPE" = "flux1-schnell" ]; then \
      wget -O models/unet/flux1-schnell.safetensors https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors && \
      wget -O models/clip/clip_l.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors && \
      wget -O models/clip/t5xxl_fp8_e4m3fn.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors && \
      wget -O models/vae/ae.safetensors https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors; \
    elif [ "$MODEL_TYPE" = "flux1-dev" ]; then \
      wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/unet/flux1-dev.safetensors https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors && \
      wget -O models/clip/clip_l.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors && \
      wget -O models/clip/t5xxl_fp8_e4m3fn.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors && \
      wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/vae/ae.safetensors https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors; \
    fi

# Stage 3: Final image
FROM base as final

# Copy models from stage 2 to the final image
COPY --from=downloader /comfyui/models /comfyui/models

# Start container
CMD ["/start.sh"]
