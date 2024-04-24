# Image: jthet/rapper-llm
FROM python:3.11


RUN apt-get update && apt-get install -y libhdf5-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /home

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./rapper-llm

CMD ["jupyter-notebook","--ip", "0.0.0.0","--allow-root"]

# COPY models ./models
# COPY api.py ./api.py


# CMD ["sleep", "infinity"]
# CMD ["python3", "api.py"]


# COMMANDS:
# docker run -d -v $(pwd):/home/rapper-llm --name=app jthet/rapper-llm:latest sleep infinity
# docker exec -it app /bin/bash