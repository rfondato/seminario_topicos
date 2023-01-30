
from airflow.models import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.contrib.sensors.file_sensor import FileSensor
from datetime import datetime

default_args = {
    'owner': 'rFondato', 
    'retries': 0,
    'start_date': datetime(2023, 1, 22)
    }
with DAG('sample-data', 
        default_args=default_args,
        max_active_runs=1,
        catchup=False,
        schedule_interval='*/5 * * * *') as dag:
    model_created_sensor = FileSensor(
        task_id='check_model',
        filepath='/model/.created'
    )

    sample = BashOperator(
        task_id='sample',
        bash_command="/opt/spark/bin/spark-submit --master 'spark://master:7077' --conf spark.executor.memory=1g --conf spark.executor.cores=2 --conf spark.cores.max=2 --conf spark.sql.shuffle.partitions=8 /app/sample.py"
    )

    model_created_sensor >> sample
