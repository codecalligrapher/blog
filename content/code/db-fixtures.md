---
title: "Databricks Fixtures for PyTest"
date: 2023-08-18T00:00:00-00:00
ShowCodeCopyButtons: true
hideSummary: true
summary: false
tags: ["pyspark", "databricks", "python"]
comments: true
showReadingTime: false
---

This creates fixtures for running tests as jobs in Databricks. This does *NOT* use the `yield` keyword since the fixtures load the existing context natively

```python

import pytest

@pytest.fixture
def spark():
  import pyspark
  from pyspark import SparkContext
  from pyspark.sql import SparkSession

  sc = SparkContext.getOrCreate()
  spark = SparkSession()

  return spark

def some_test(spark):
  from pyspark.dbutils import DBUtils

  dbutils = DBUtils(spark)
  # test
```