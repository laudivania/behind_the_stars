FROM python:3.12.9
COPY behind_the_stars /behind_the_stars
COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD uvicorn behind_the_stars.api.fast:app --host 0.0.0.0 --port $PORT
