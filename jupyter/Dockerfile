FROM rfondato/spark:3.3.1

WORKDIR /notebook

RUN pip3 install --no-cache-dir --default-timeout=120 -r /tmp/requirements.txt && \
  rm -rvf /tmp/requirements.txt

COPY requirements.txt /tmp/

CMD [ "/usr/local/bin/jupyter", "notebook", "--allow-root", "--no-browser", "--ip=0.0.0.0"]

EXPOSE 8888
