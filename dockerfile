FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel
ENV PYTHONUNBUFFERED 1
WORKDIR /app

ADD . /app/
RUN python -m pip install Django
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000
