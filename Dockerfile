FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install -r requirements.txt
COPY . .
EXPOSE 8080 8501
CMD ["uvicorn", "truefield.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
