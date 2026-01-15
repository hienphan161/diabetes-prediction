FROM python:3.9

# Create a folder /app if it doesn't exist,
# the /app folder is the current working directory
WORKDIR /app

# Copy requirements first for better caching
COPY ./requirements.txt /app

# Copy model files
COPY ./results/4_best_model /app/models

# Copy API and utils files
COPY ./src/api/main.py /app/main.py
COPY ./src/utils.py /app/utils.py

# Set environment variables
ENV MODEL_PATH=/app/models/best_model.joblib
ENV METRICS_PORT=8099

EXPOSE 30000
EXPOSE 8099

# Disable pip cache to shrink the image size a little bit
RUN pip install -r requirements.txt --no-cache-dir

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "30000"]

