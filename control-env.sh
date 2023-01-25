#!/usr/bin/env bash

function stop {
  echo "Stopping and removing containers"
  docker-compose --project-name tp-seminario-rodrigo down
}

function cleanup {
  echo "Removing volume"
  docker volume rm tp-seminario-rodrigo_postgres-airflow-data
}

function start {
  echo "Starting up"
  docker-compose --project-name tp-seminario-rodrigo up -d
}

function createImages {
  echo "Creating custom local images"
  echo "Creating rfondato/pyspark:3.3.1 ..."
  docker build ./pyspark/ -f ./pyspark/Dockerfile -t rfondato/pyspark:3.3.1
  echo "Creating rfondato/airflow-spark ..."
  docker build ./airflow/ -f ./airflow/Dockerfile -t rfondato/airflow-spark
  echo "Creating rfondato/jupyter ..."
  docker build ./jupyter/ -f ./jupyter/Dockerfile -t rfondato/jupyter
  echo "Creating rfondato/mlflow"
  docker build ./mlflow/ -f ./mlflow/Dockerfile -t rfondato/mlflow
  echo "Finished creating images. Please run './control-env.sh start' to run the containers"
}

function update {
  echo "Updating code ..."
  git pull --all

  echo "Updating docker images ..."
  docker-compose --project-name tp-seminario-rodrigo pull

  echo "You probably should restart"
}

function info {
  echo '
  Everything is ready, access your host to learn more (ie: http://localhost/)
  '
}

function token {
  echo 'Your TOKEN for Jupyter Notebook is:'
  SERVER=$(docker exec -it jupyter jupyter notebook list)
  echo "${SERVER}" | grep '/notebook' | sed -E 's/^.*=([a-z0-9]+).*$/\1/'
}

case $1 in
  start )
  start
  info
    ;;

  stop )
  stop
    ;;

  create-images )
  createImages
    ;;

  cleanup )
  stop
  cleanup
    ;;

  update )
  update
    ;;

  logs )
  docker-compose --project-name tp-seminario-rodrigo logs -f
    ;;

  token )
  token
    ;;

  * )
  printf "ERROR: Missing command\n  Usage: `basename $0` (start|stop|cleanup|token|logs|update|create-images)\n"
  exit 1
    ;;
esac
