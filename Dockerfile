FROM python:3.11.9-slim

WORKDIR /dash

COPY . /dash

RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r /dash/requirements.txt

CMD ["python","app.py"]
