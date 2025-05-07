# Databricks notebook source
# DBTITLE 1,BigData Final Project
import requests
import json

def get_data_from_api(api_url, params=None, headers=None):
   
    try:
        response = requests.get(api_url, params=params, headers=headers)
        response.raise_for_status()  
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from API: {e}")
        return None

api_url = "https://data.montgomerycountymd.gov/api/views/mmzv-x632/rows.json?accessType=DOWNLOAD"  
data = get_data_from_api(api_url)

# COMMAND ----------

import pandas as pd
data_rows = data.get('data', [])
c=data.get('meta').get('view').get('columns')
col=[col['name'] for col in c]
df = pd.DataFrame(data_rows, columns=col)
df=df.iloc[:,9:]
df.head()

# COMMAND ----------

df.info()

# COMMAND ----------

!pip install pymongo
import pymongo
import pandas as pd
from pymongo import MongoClient

# COMMAND ----------

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from pymongo.errors import AutoReconnect

uri = "mongodb+srv://hello:kitty123@cluster0.hqc9i.mongodb.net/?retryWrites=true&w=majority&connectTimeoutMS=300000"
client = MongoClient(uri, server_api=ServerApi('1'))
data = df.to_dict(orient='records')


db = client['Project']
collection = db['CrashData']

try:
    batch_size = 1000
    for i in range(0, 120000, batch_size):
        batch = data[i : i + batch_size]
        collection.insert_many(batch, ordered=False)
except AutoReconnect as e:
    print(f"AutoReconnect error: {e}")


# COMMAND ----------

# MAGIC %md
# MAGIC ### Bronze Layer

# COMMAND ----------

data = pd.DataFrame(list(collection.find()))
data.info()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Analysis

# COMMAND ----------

Required = [
    'Crash Date/Time', 
    'Road Name', 
    'Collision Type', 
    'Weather', 
    'Surface Condition', 
    'Injury Severity', 
    'Vehicle Body Type', 
    'Speed Limit', 
    'Driver At Fault', 
    'Traffic Control', 
    'Latitude', 
    'Longitude'
]
df_required = data[Required]
df_required.head()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Data Cleaning Silver Layer

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when

spark = SparkSession.builder.appName("CrashData").getOrCreate()
spark_df = spark.createDataFrame(df_required)
spark_df.show()
spark_df = (
    spark_df
    .withColumn("Weather", when(col("Weather").isNull(), "Unknown").otherwise(col("Weather")))
    .withColumn("Surface Condition", when(col("Surface Condition").isNull(), "Unknown").otherwise(col("Surface Condition")))
    .withColumn("Injury Severity", when(col("Injury Severity").isNull(), "No Injury").otherwise(col("Injury Severity")))
)
spark_df = spark_df.dropDuplicates()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Aggregations - Gold Layer

# COMMAND ----------

# Crash Count by Weather Condition

spark_df.groupBy("Weather").count().orderBy("count", ascending=False).show()



# COMMAND ----------

# Average Injury Severity by Vehicle Body Type
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import col, count, avg, hour, desc

severity_index = StringIndexer(inputCol="Injury Severity", outputCol="Severity_Index").fit(spark_df)
df_severity = severity_index.transform(spark_df)
df_severity.groupBy("Vehicle Body Type").agg(avg("Severity_Index").alias("Avg_Severity")).show()


# COMMAND ----------

# Top 5 Collision Types Leading to Injuries
spark_df.groupBy("Collision Type").agg(count("*").alias("Total")) \
  .orderBy(col("Total").desc()).limit(5).show()


# COMMAND ----------

# Crash Density by Speed Limit
spark_df.groupBy("Speed Limit").count().orderBy("Speed Limit").show()


# COMMAND ----------

# Crash Count by Traffic Control Type
spark_df.groupBy("Traffic Control").count().orderBy(col("count").desc()).show()


# COMMAND ----------

# Crash Trends by Hour of Day
from pyspark.sql.functions import hour
spark_df.withColumn("Hour", hour("Crash Date/Time")).groupBy("Hour").count().orderBy("Hour").show()


# COMMAND ----------

# MAGIC %md
# MAGIC ### Visualizations

# COMMAND ----------

df_pd=spark_df.toPandas()

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(data=df_pd, x="Weather")
plt.title("Crash Count by Weather")


# COMMAND ----------

df_pd["Driver At Fault"].value_counts().plot.pie(autopct="%1.1f%%")
plt.title("Driver At Fault Distribution")



# COMMAND ----------

sns.scatterplot(data=df_pd, x="Longitude", y="Latitude", hue="Injury Severity")
plt.title("Crash Map by Severity")



# COMMAND ----------

hourly_df = spark_df.withColumn("Hour", hour("Crash Date/Time")) \
              .groupBy("Hour").agg(count("*").alias("count")) \
              .orderBy("Hour")

hourly_df_pd = hourly_df.toPandas()


sns.lineplot(x="Hour", y="count", data=hourly_df_pd)
plt.title("Crash Frequency by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Crash Count")
plt.grid(True)
plt.tight_layout()
plt.show()


# COMMAND ----------

collision_weather_agg = spark_df.groupBy("Weather", "Collision Type") \
                          .agg(count("*").alias("Crash Count"))
collision_weather_df = collision_weather_agg.toPandas()

collision_weather_pivot = collision_weather_df.pivot_table(
    index="Weather",
    columns="Collision Type",
    values="Crash Count",
    aggfunc="sum",
    fill_value=0
)
collision_weather_pivot.plot(
    kind='bar',
    stacked=True,
    figsize=(12, 6),
    cmap='Set3'
)

plt.title("Crashes by Collision Type and Weather")
plt.xlabel("Weather Condition")
plt.ylabel("Crash Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Regression – Predict Injury Severity

# COMMAND ----------

from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
df = spark_df.withColumn("Speed Limit", col("Speed Limit").cast("int"))

df_reg = df.select("Injury Severity", "Weather", "Collision Type", "Vehicle Body Type", "Speed Limit").dropna()
indexers = [
    StringIndexer(inputCol="Injury Severity", outputCol="label"),
    StringIndexer(inputCol="Weather", outputCol="Weather_index"),
    StringIndexer(inputCol="Collision Type", outputCol="Collision_index"),
    StringIndexer(inputCol="Vehicle Body Type", outputCol="Vehicle_index")
]

assembler = VectorAssembler(
    inputCols=["Weather_index", "Collision_index", "Vehicle_index", "Speed Limit"],
    outputCol="features"
)

lr = LinearRegression(labelCol="label", featuresCol="features")
pipeline = Pipeline(stages=indexers + [assembler, lr])
model = pipeline.fit(df_reg)
predictions = model.transform(df_reg)

# Evaluate
from pyspark.ml.evaluation import RegressionEvaluator
r2 = RegressionEvaluator(metricName="r2", labelCol="label", predictionCol="prediction").evaluate(predictions)
print(f"Injury Severity Regression R² Score: {r2:.2f}")


# COMMAND ----------

# MAGIC %md
# MAGIC #### Clustering – Crash Hotspot Clustering

# COMMAND ----------

from pyspark.ml.clustering import KMeans

df = df.withColumn("Longitude", col("Longitude").cast("double"))
df = df.withColumn("Latitude", col("Latitude").cast("double")) 
severity_indexer = StringIndexer(inputCol="Injury Severity", outputCol="Severity_Index")
df_cluster = severity_indexer.fit(df).transform(df)

# Drop nulls and assemble features
df_cluster = df_cluster.select("Latitude", "Longitude", "Speed Limit", "Severity_Index").dropna()

vec_assembler = VectorAssembler(inputCols=["Latitude", "Longitude", "Speed Limit", "Severity_Index"], outputCol="features")
df_features = vec_assembler.transform(df_cluster)

kmeans = KMeans(k=4, seed=1)
model = kmeans.fit(df_features)
clusters = model.transform(df_features)

# Evaluate
from pyspark.ml.evaluation import ClusteringEvaluator
silhouette = ClusteringEvaluator().evaluate(clusters)
print(f"Silhouette Score (Clustering Quality): {silhouette:.2f}")
