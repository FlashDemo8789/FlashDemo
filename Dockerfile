FROM python:3.10-slim

# Install system packages if needed
RUN apt-get update && apt-get install -y \
    git \
    wkhtmltopdf \
    libxrender1 \
    libxext6 \
    libfontconfig1 \
    && apt-get clean

WORKDIR /app

# Copy your requirements
COPY requirements.txt /app/requirements.txt

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install pinned dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your project
COPY . /app

# Expose port if using streamlit
EXPOSE 8501

CMD ["streamlit", "run", "analysis_flow.py", "--server.address=0.0.0.0"]