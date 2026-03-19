FROM tensorflow/tensorflow:2.20.0
COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip
RUN pip uninstall transformers
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt
COPY behind_the_stars /behind_the_stars
COPY raw_data/pickle_embeddings.pkl /raw_data/pickle_embeddings.pkl
CMD uvicorn behind_the_stars.api.fast:app --host 0.0.0.0 --port $PORT
