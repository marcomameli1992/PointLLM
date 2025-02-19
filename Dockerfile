# Utilizza un'immagine base con supporto CUDA
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Imposta la directory di lavoro principale
WORKDIR /app

# Installa pacchetti di sistema necessari
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-venv \
    git \
    curl \
    ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Installa Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# Clona il repository PointLLM
#RUN git clone https://github.com/marcomameli1992/PointLLM.git /app/PointLLM

# Imposta la directory di lavoro corretta
WORKDIR /app/PointLLM

# Verifica che il file pyproject.toml esista
RUN ls -l /app/PointLLM

COPY pyproject.toml ./

# Installa le dipendenze usando Poetry
RUN poetry install --no-root
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -e .
RUN python3 -m pip install objaverse
RUN python3 -m pip install transformers
RUN python3 -m pip install gradio

# Imposta il PYTHONPATH e avvia il comando desiderato
# comando per chat da terminale -> mancano i dati
#CMD PYTHONPATH=$PWD poetry run python pointllm/eval/PointLLM_chat.py \
#    --model_name RunsenXu/PointLLM_7B_v1.2 \
#    --data_name data/objaverse_data \
#    --torch_dtype float16 #32 finisce la memoria perchè richiede 24GB liberi noi ne abbiamo 23.7

# Gradio app
# non si può scegliere il torch_dtype e non bastano 24GB di memoria
CMD PYTHONPATH=$PWD python3 pointllm/eval/chat_gradio.py \
    --model-name RunsenXu/PointLLM_7B_v1.2 \
    --data_path data/objaverse_data