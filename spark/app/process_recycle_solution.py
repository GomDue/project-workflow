'''
target이 기존 데이터베이스에 없는 name, tag, category를 확인하고 없다면 추가

target : 
RecycleSolution.csv
    name : 분비배출 대상
    material : 재질(공백으로 구분)
    tag : 태그(#으로 구분)
    solution : 분리배출 방법
    imgUrl : 이미지 URL


exist : 
    pass

Doesn't exist :
    INSERT INTO waste
    VALUES (created_date, modified_date, image_url, name, solution, state, writer_id) 
        created_date = datetime.now()
        modified_date = datetime.now()
        image_url = imgUrl
        name = name
        solution = solution
        state = 1
        writer_id = 1(admin)

    INSERT INTO tag
    VALUES (name, waste_id)
        name = tag
        waste_id = waste_id
    
    INSERT INTO category
    VALUES (name, waste_id)
        name = material
        waste_id = waste_id

'''
import datetime

from airflow.models import Variable

from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, col, split, explode

# config AWS RDS
AWS_RDS_URL = "jdbc:mysql://{0}:{1}/{2}".format(Variable.get('aws_rds_host'), 3306, Variable.get('aws_rds_database'))
AWS_RDS_USER = Variable.get('aws_rds_user')
AWS_RDS_PASSWORD = Variable.get('aws_rds_password')

recycle_solution = ""

# Create Spark Session
spark = SparkSession.builder.appName("RecycleSolution").getOrCreate()

# JDBC Reader Settings
reader = (spark.read.format("jdbc")
        .option("url", AWS_RDS_URL)
        .option('driver', 'com.mysql.cj.jdbc.Driver')
        .option("user", AWS_RDS_USER)
        .option("password", AWS_RDS_PASSWORD))

# Load CSV file
rs_df = (spark.read.format("csv")
        .option("header", "true")
        .option("nullValue", "")
        .load(recycle_solution))

# Filter valid data
filtered_df = (rs_df
        .filter(col("solution").isNotNull())
        .filter(col("tag").isNotNull())
        .filter(col("material").isNotNull()))

# Load existing data from RDS
waste_rds_df = reader.option("dbtable", "waste").load()

# Check and insert new waste data
new_waste_df = (filtered_df
        .join(waste_rds_df, on="name", how="left_anti"))
new_waste_df.show()

if new_waste_df.count() > 0:
    new_waste_insert_df = (new_waste_df
        .withColumn("created_date",     lit(datetime.datetime.now()))
        .withColumn("modified_date",    lit(datetime.datetime.now()))
        .withColumn("state",            lit(1))
        .withColumn("writer_id",        lit(1))
        .select("created_date", "modified_date", col("imgUrl").alias("image_url"), "name", "solution", "state", "writer_id"))
    
    new_waste_insert_df.write.format("jdbc") \
        .option("url", AWS_RDS_URL) \
        .option("dbtable", "waste") \
        .option("user", AWS_RDS_USER) \
        .option("password", AWS_RDS_PASSWORD) \
        .option("driver", "com.mysql.cj.jdbc.Driver") \
        .mode("append") \
        .save()

# Reload updated waste data
waste_rds_df = reader.option("dbtable", "waste").load()

# Load existing tag and category data
tag_rds_df = reader.option("dbtable", "tag").load().select(col("waste_id").alias("id"), col("name").alias("tag"))
category_rds_df = reader.option("dbtable", "category").load().select(col("waste_id").alias("id"), col("name").alias("material"))

# Explode and insert new tag data
new_tags_df = (filtered_df
        .withColumn("tag", explode(split("tag", "#")))
        .filter(col("tag") != "")
        .join(waste_rds_df, on="name")
        .join(tag_rds_df, on=["id", "tag"], how="left_anti")
        .select(col("id").alias("waste_id"), col("tag").alias("name")))
new_tags_df.show()

if new_tags_df.count() > 0:
    new_tags_df.write.format("jdbc") \
        .option("url", AWS_RDS_URL) \
        .option("dbtable", "tag") \
        .option("user", AWS_RDS_USER) \
        .option("password", AWS_RDS_PASSWORD) \
        .option("driver", "com.mysql.cj.jdbc.Driver") \
        .mode("append") \
        .save()

# Explode and insert new category data
new_categories_df = (filtered_df
        .withColumn("material", explode(split("material", " ")))
        .join(waste_rds_df, on="name")
        .join(category_rds_df, on=["id", "material"], how="left_anti")
        .select(col("id").alias("waste_id"), col("material").alias("name")))
new_categories_df.show()

if new_categories_df.count() > 0:
    new_categories_df.write.format("jdbc") \
        .option("url", AWS_RDS_URL) \
        .option("dbtable", "category") \
        .option("user", AWS_RDS_USER) \
        .option("password", AWS_RDS_PASSWORD) \
        .option("driver", "com.mysql.cj.jdbc.Driver") \
        .mode("append") \
        .save()