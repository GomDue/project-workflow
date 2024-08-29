import os
from datetime import datetime

from airflow.models import Variable

NOW_DATE = datetime.now().strftime('%y%m%d')

# Configurations for AWS RDS
class AWSRDSConfig:
    AWS_RDS_URL = "jdbc:mysql://{0}:{1}/{2}".format(
        Variable.get("aws_rds_host"), 3306, Variable.get("aws_rds_database")
    )
    AWS_RDS_USER = Variable.get("aws_rds_user")
    AWS_RDS_PASSWORD = Variable.get("aws_rds_password")
    AWS_RDS_DRIVER = "com.mysql.cj.jdbc.Driver"


# Configurations for Google Cloud Platform (GCP)
class GCPConfig:
    GOOGLE_CONN = "google_cloud_default"

    GOOGLE_SHEET_ID = "1O6ryWUCco2_tOlzvtcPCs3eDTPnIQkKDH2955tCUdxg"
    GOOGLE_SHEET_NAME = "시트1"


# Configurations for Airflow
class AirflowConfig:
    AIRFLOW_DIR = os.path.abspath("./airflow")


# Configurations for PySpark
class PysparkConfig(AirflowConfig):
    PYSPARK_DIR = os.path.join(AirflowConfig.AIRFLOW_DIR, "spark", "app")
    PYSPARK_CONN = "spark_default"
    PYSPARK_CONF = {"spark.master":"spark://spark:7077"}

    PYSPARK_APP__RECYCLE_SOLUTION_PATH = os.path.join(PYSPARK_DIR, 'process_recycle_solution.py')
    

# Configurations for Recycle Solution
class RecycleSolutionConfig(AirflowConfig):
    RECYCLE_SOLUTION_DIR = os.path.join(AirflowConfig.AIRFLOW_DIR, "data", "solutions")
    RECYCLE_SOLUTION_FILE_NAME = f"recycle_solution_{NOW_DATE}.csv"
    RECYCLE_SOLUTION_FILE_PATH = os.path.join(RECYCLE_SOLUTION_DIR, RECYCLE_SOLUTION_FILE_NAME)
    
# Main configuration class to access specific configurations
class BaseConfig:
    def __getitem__(self, config_type):
        config_table = {
            "AWSRDSConfig"          : AWSRDSConfig,
            "GCPConfig"             : GCPConfig,
            "AirflowConfig"         : AirflowConfig,
            "PysparkConfig"         : PysparkConfig,
            "RecycleSolutionConfig" : RecycleSolutionConfig
        }
        return config_table.get(config_type, None)

CONFIG = BaseConfig()