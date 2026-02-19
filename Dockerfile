FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip

RUN pip install --default-timeout=1000 \
    torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --default-timeout=1000 -r requirements.txt

CMD ["python", "app.py"]
