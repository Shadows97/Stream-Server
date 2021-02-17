# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.7-slim-buster

EXPOSE 5000

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

ENV CAMERA=obdet

RUN apt-get update ##[edited]
RUN apt-get install ffmpeg libsm6 libxext6  -y

# Install pip requirements
COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt

RUN pip3 install --user --upgrade tensorflow

WORKDIR /app
COPY . /app

# # Switching to a non-root user, please refer to https://aka.ms/vscode-docker-python-user-rights
# RUN useradd appuser && chown -R appuser /app
# USER appuser

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["python3", "app.py"]
