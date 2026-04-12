FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install dependencies (without torch in requirements.txt)
RUN pip install --no-cache-dir -r requirements.txt

# Install CPU-only torch explicitly
RUN pip install --no-cache-dir torch==2.11.0 -f https://download.pytorch.org/whl/cpu

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
