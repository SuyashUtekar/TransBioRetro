# Use official lightweight Python image
FROM python:3.10-slim

# Expose port 8501 (default Streamlit port)
EXPOSE 8501

# Set working directory
WORKDIR /app

# Copy requirements files
COPY requirements.txt ./requirements.txt

# Install system dependencies for RDKit and graphviz (important)
RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    curl \
    libxrender1 \
    libxext6 \
    libsm6 \
    libglib2.0-0 \
    graphviz \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy your entire app code into the container
COPY . .

# Run the Streamlit app
ENTRYPOINT ["streamlit", "run", "net_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
