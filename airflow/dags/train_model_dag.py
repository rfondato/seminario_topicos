
from airflow.models import DAG
from airflow.operators.bash_operator import BashOperator
from datetime import datetime

default_args = {
    'owner': 'rFondato', 
    'retries': 0,
    'start_date': datetime(2023, 1, 22)
    }
with DAG('train-model', 
          default_args=default_args,
          max_active_runs=1,
          catchup=False,
          schedule_interval='0 0 * * *') as dag:
    partition_data = BashOperator(
        task_id='partition_data',
        bash_command="/opt/spark/bin/spark-submit --master 'spark://master:7077' --conf spark.executor.memory=1g /app/partition.py"
    )
    train_model = BashOperator(
        task_id='train_model',
        bash_command="/opt/spark/bin/spark-submit --master 'spark://master:7077' --conf spark.executor.memory=1g /app/train.py"
    )
    test_model = BashOperator(
        task_id='test_model',
        bash_command="/opt/spark/bin/spark-submit --master 'spark://master:7077' --conf spark.executor.memory=1g /app/test.py"
    )

    partition_data >> train_model >> test_model
