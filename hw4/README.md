hw4 files in S3 bucket de300spring2024: 
/cheryl_chen/hw4/
├── dags/
│   ├── eda_workflow.py: contains the Airflow DAG definition
│   └── utils.py: utility functions called by the DAG
├── scripts/
│   ├── load_data.py: loads data
│   ├── eda_standard.py: performs standard data cleaning
│   ├── eda_spark.py: performs spark data cleaning
│   ├── fe_standard.py: standard feature engineering
│   ├── fe_spark.py: spark feature engineering
│   └── train_model_standard.py: standard ML
│   └── train_model_spark.py: spark ML
│   └── select_best_model.py: selects the best model out of standard and spark models combined
├── config/
│   └── config.toml
├── requirements.txt
