# Image: jthet/hurricane-prediction
FROM python:3.11

RUN apt-get update && apt-get install -y libhdf5-dev

RUN pip install tensorflow==2.15
RUN pip install Flask==3.0
RUN pip install numpy==1.26
RUN pip install pillow


WORKDIR /home

COPY models ./models
COPY api.py ./api.py


CMD ["python3", "api.py"]