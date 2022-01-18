FROM python:3.8-slim

RUN apt update && \
apt install --no-install-recommends -y build-essential gcc && \
apt install -y git && \
apt clean && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/samwaltonnorwood/dtumlops.git
WORKDIR "/dtumlops"

RUN pip install torch==1.10.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html --no-cache-dir && \
pip install -e . --no-cache-dir && \
pip install -r requirements.txt --no-cache-dir

ENTRYPOINT ["python", "-u", "src/main.py"]
