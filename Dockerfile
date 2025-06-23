FROM nvidia/cuda:12.4.0-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
      python3.10 \
      python3.10-dev \
      python3.10-distutils \
      git wget curl build-essential gcc g++ cmake dos2unix \
      wkhtmltopdf && \
    rm -rf /var/lib/apt/lists/*
    
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.10 get-pip.py && \
    rm get-pip.py

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124 --timeout 600 --retries 10 --resume-retries 5
RUN pip install --no-cache-dir --timeout 500 --retries 10 --resume-retries 5 -r requirements.txt

RUN mkdir -p /app/model_cache
ENV TRANSFORMERS_CACHE=/app/model_cache

RUN python3.10 -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
model_name='TheBloke/Mistral-7B-Instruct-v0.2-GPTQ'; \
AutoTokenizer.from_pretrained(model_name); \
AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')"

COPY . .

RUN dos2unix start.sh && chmod +x start.sh

RUN apt-get purge -y dos2unix && apt-get autoremove -y

EXPOSE 8080

CMD ["./start.sh"]