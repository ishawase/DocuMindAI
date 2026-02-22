FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip

RUN pip install --default-timeout=1000 \
    torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --default-timeout=1000 -r requirements.txt

RUN pip install --upgrade "huggingface_hub>=0.23.0"

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

