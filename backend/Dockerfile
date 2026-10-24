FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py superkart_model.joblib ./
EXPOSE 7860
CMD ["gunicorn", "--workers", "2", "--bind", "0.0.0.0:7860", "app:superkart_api"]
