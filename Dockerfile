FROM python:3.10-bullseye

ENV HOST 0.0.0.0
ENV PORT 8080

WORKDIR /app
COPY . .

RUN apt update && apt install -y python3-opencv
RUN pip install --no-cache-dir -r requirements.txt

CMD uvicorn main:app --host $HOST --port $PORT
