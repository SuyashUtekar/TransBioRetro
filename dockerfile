# Use official lightweight Python image
FROM python:3.10-slim

# Install system dependencies for RDKit and graphviz (important)
RUN apt-get update && apt-get install -y \
    build-essential \
    libxrender1 \
    libxext6 \
    libsm6 \
    libglib2.0-0 \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements files and install Python dependencies
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy your entire app code into the container
COPY . .

# Expose port 8501 (default Streamlit port)
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "net_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
