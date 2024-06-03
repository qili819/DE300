from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from utils import load_data, eda_standard, eda_spark, fe_standard, fe_spark, train_model_standard, train_model_spark, select_best_model

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
}

with DAG(
    'eda_workflow',
    default_args=default_args,
    description='A workflow for EDA and feature engineering',
    schedule_interval='@daily',
    start_date=days_ago(1),
    catchup=False,
) as dag:

    load_data_task = PythonOperator(
        task_id='load_data',
        python_callable=load_data,
    )

    eda_standard_task = PythonOperator(
        task_id='eda_standard',
        python_callable=eda_standard,
    )

    eda_spark_task = PythonOperator(
        task_id='eda_spark',
        python_callable=eda_spark,
    )

    fe_standard_task = PythonOperator(
        task_id='fe_standard',
        python_callable=fe_standard,
    )

    fe_spark_task = PythonOperator(
        task_id='fe_spark',
        python_callable=fe_spark,
    )

    train_model_standard_task = PythonOperator(
        task_id='train_model_standard',
        python_callable=train_model_standard,
    )

    train_model_spark_task = PythonOperator(
        task_id='train_model_spark',
        python_callable=train_model_spark,
    )

    select_best_model_task = PythonOperator(
        task_id='select_best_model',
        python_callable=select_best_model,
    )

    load_data_task >> [eda_standard_task, eda_spark_task]
    eda_standard_task >> fe_standard_task
    eda_spark_task >> fe_spark_task
    fe_standard_task >> train_model_standard_task
    fe_spark_task >> train_model_spark_task
    [train_model_standard_task, train_model_spark_task] >> select_best_model_task
