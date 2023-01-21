FROM openjdk:11-jre-slim

WORKDIR /app

ARG SPARK_VERSION=3.3.1

ARG HADOOP_VERSION=3

ENV SPARK_HOME /opt/spark

# Install java + spark
RUN apt-get update && \
  apt-get install -y wget ca-certificates procps && \
  wget https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz -O - | tar zx -C /opt && \
  ln -s /opt/spark-${SPARK_VERSION}-bin-hadoop{HADOOP_VERSION} /opt/spark && \
  echo "\nexport PATH=\${PATH}:/opt/spark/bin" >> /etc/bash.bashrc && \
  echo "\nexport SPARK_NO_DAEMONIZE=1" >> /etc/bash.bashrc && \
  rm -rf /var/lib/apt/lists/*

# Install everything needed for pyspark
RUN apt-get update && \
  apt-get --no-install-recommends --no-install-suggests install -y \
  python3 python3-pip python3-setuptools python3-distutils && \
  update-alternatives --install /usr/bin/python python /usr/bin/python3 10 && \
  pip3 install --no-cache-dir --default-timeout=120 -r /tmp/requirements.txt && \
  pip install pyspark==${SPARK_VERSION}} \
  apt-get autoremove -y && \
  rm -rvf /tmp/requirements.txt /var/lib/apt/lists/*

COPY requirements.txt /tmp/

CMD ["/opt/spark/bin/spark-shell"]

EXPOSE 4040 6066 7077 8080
