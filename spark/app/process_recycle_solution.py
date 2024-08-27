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
AWS_RDS_DB = Variable.get('aws_rds_database')
AWS_RDS_USER = Variable.get('aws_rds_user')
AWS_RDS_PASSWORD = Variable.get('aws_rds_password')

recycle_solution = "/home/user/airflow/data/solutions/recycle_solution_240827.csv"

# Create Spark Session
spark = (SparkSession.builder.appName("RecycleSolution").getOrCreate())

# JDBC Reader Settings
reader = (spark
        .read
        .format("jdbc")
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
filterd_df = (rs_df
        .filter(col("solution").isNotNull())
        .filter(col("tag").isNotNull())
        .filter(col("material").isNotNull()))


# Find and insert newly added data
new_waste_df = (filterd_df
        .join(reader.option("dbtable", "waste").load(), on="name", how="outer")
        .filter(col("id").isNull()))

not_in_rds_list = list(new_waste_df.select("name").toPandas()["name"].unique())

if new_waste_df.count() > 0:
    new_waste_insert_df = (rs_df
        .filter(col("name").isin(not_in_rds_list))
        .withColumn("created_date",     lit(datetime.datetime.now()))
        .withColumn("modified_date",    lit(datetime.datetime.now()))
        .withColumn("image_url",        rs_df["imgUrl"])
        .withColumn("name",             rs_df["name"])
        .withColumn("solution",         rs_df["solution"])
        .withColumn("state",            lit(1))
        .withColumn("writer_id",        lit(1))
        .select("created_date", "modified_date", "image_url", "name", "solution", "state", "writer_id"))
    
    (new_waste_insert_df
        .write
        .format("jdbc")
        .option("url", AWS_RDS_URL)
        .option("dbtable", "waste")
        .option("user", AWS_RDS_USER)
        .option("password", AWS_RDS_PASSWORD)
        .option("driver", "com.mysql.cj.jdbc.Driver")
        .mode("append")
        .save())
    

# Import existing waste, tag, category tables
waste_rds_df = (reader
        .option("dbtable", "waste")
        .load())
tag_rds_df = (reader
        .option("dbtable", "tag")
        .load()
        .select(col("waste_id").alias("id"), col("name").alias("tag")))
category_rds_df = (reader
        .option("dbtable", "category")
        .load()
        .select(col("waste_id").alias("id"), col("name").alias("category")))
joined_df = (waste_rds_df
        .join(tag_rds_df, on="id", how="inner")
        .join(category_rds_df, on="id", how="inner"))


# Check and insert new tag data
not_in_tag_df = (filterd_df
        .withColumn("tag", explode(split("tag", "#")))
        .filter(col("tag") != "")
        .select("name", "tag")
        .join(joined_df, on=["name", "tag"], how="left_anti")
        .join(waste_rds_df.select("id", "name"), on="name", how="left")
        .filter(col("id").isNotNull()))

if not_in_tag_df.count() > 0:
    (not_in_tag_df
        .select(col("id").alias("waste_id"), col("tag").alias("name"))
        .write
        .format("jdbc")
        .option("url", AWS_RDS_URL)
        .option("dbtable", "tag")
        .option("user", AWS_RDS_USER)
        .option("password", AWS_RDS_PASSWORD)
        .option("driver", "com.mysql.cj.jdbc.Driver")
        .mode("append")
        .save())


# Check and insert new category data
not_in_category_df = (filterd_df
        .withColumn("material", explode(split("material", " ")))
        .select("name", col("material").alias("category"))
        .join(joined_df, on=["name", "category"], how="left_anti")
        .join(waste_rds_df.select("id", "name"), on="name", how="left")
        .filter(col("id").isNotNull()))

if filterd_df.count() > 0:
    (not_in_category_df
        .select(col("id").alias("waste_id"), col("category").alias("name"))
        .write
        .format("jdbc")
        .option("url", AWS_RDS_URL)
        .option("dbtable", "category")
        .option("user", AWS_RDS_USER)
        .option("password", AWS_RDS_PASSWORD)
        .option("driver", "com.mysql.cj.jdbc.Driver")
        .mode("append")
        .save())
