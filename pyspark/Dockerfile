FROM python:3.7-slim-buster

WORKDIR /app

ARG SPARK_VERSION=3.3.1

ARG HADOOP_VERSION=3

ENV SPARK_HOME /opt/spark

# Install spark
RUN apt-get update && \
  apt install -y ca-certificates procps openjdk-11-jre-headless scala wget && \
  export JAVA_HOME='/usr/lib/jvm/java-11-openjdk-amd64' && \
  export PATH=$PATH:$JAVA_HOME/bin && \
  wget https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz -O - | tar zx -C /opt && \
  mv /opt/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}/ /opt/spark/ && \
  echo "\nexport PATH=\${PATH}:/opt/spark/bin" >> /etc/bash.bashrc && \
  echo "\nexport SPARK_NO_DAEMONIZE=1" >> /etc/bash.bashrc && \
  rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/

# Install everything needed for pyspark
RUN pip install --no-cache-dir --default-timeout=120 -r /tmp/requirements.txt && \
  pip install pyspark==${SPARK_VERSION} && \
  rm -rvf /tmp/requirements.txt

CMD ["/opt/spark/bin/spark-shell"]

EXPOSE 4040 6066 7077 8080
