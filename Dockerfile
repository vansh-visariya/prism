FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY . .

# Expose the port HuggingFace Spaces expects
EXPOSE 7860

# Start the unified server on port 7860 (HF Spaces default)
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
