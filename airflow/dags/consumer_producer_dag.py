
from airflow.models import DAG
from airflow.operators.docker_operator import DockerOperator
from airflow.contrib.sensors.file_sensor import FileSensor
from airflow.operators.bash_operator import BashOperator
from datetime import datetime

default_args = {
    'owner': 'rFondato', 
    'retries': 0, 
    'start_date': datetime(2023, 1, 22)
    }
with DAG('create-consumer-producer', 
          default_args=default_args,
          max_active_runs=1,
          catchup=False,
          schedule_interval='*/10 * * * *') as dag:
    
    model_created_sensor = FileSensor(
        task_id='check_model',
        filepath='/model/.created'
    )

    realtime_data_available = FileSensor(
        task_id='check_realtime_data',
        filepath='/data/real_time/_SUCCESS'
    )

    create_producer = BashOperator(
        task_id='create_producer',
        bash_command="/opt/spark/bin/spark-submit --master 'spark://master:7077' --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.1 --conf spark.executor.memory=512m --conf spark.executor.cores=1 --conf spark.cores.max=1 /app/produce.py"
    )

    create_consumer = BashOperator(
        task_id='create_consumer',
        bash_command="/opt/spark/bin/spark-submit --master 'spark://master:7077' --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.1 --conf spark.executor.memory=512m --conf spark.executor.cores=1 --conf spark.cores.max=1 /app/consume.py"
    )

    model_created_sensor >> realtime_data_available >> [create_producer, create_consumer]
