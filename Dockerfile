FROM python:3.10-slim-buster


# Dependencies
COPY requirements.txt /tmp/

# Update pip and install dependencies
RUN pip install -U pip
RUN pip install -r /tmp/requirements.txt

# Copy application at the right place
COPY src /src

ENTRYPOINT ["python", "src/cli.py"]