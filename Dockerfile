FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8


# Dependencies
COPY requirements.txt /tmp/

# Update pip and install dependencies
RUN pip install -U pip
RUN pip install -r /tmp/requirements.txt

# Copy application at the right place
COPY src /src
ENV PYTHONPATH=/

ENTRYPOINT ["python", "/src/cli.py"]