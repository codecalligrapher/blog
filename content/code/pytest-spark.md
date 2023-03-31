---
title: "Setting Up Spark for PyTest"
date: 2023-03-31T00:00:00-00:00
ShowCodeCopyButtons: true
hideSummary: true
summary: false
tags: ["python", "data-science", "pyspark"]
comments: true
showReadingTime: false
---

Snippet shows how to set up and tear down a SparkSession for writing/running tests using `pytest` in Python. This prevents time from being wasted in setting up and tearing down a new `SparkSession` for each test!

```python
import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope='session')
def spark(request):
    spark = (
        SparkSession
            .builder
            .appName('appname')
            .getOrCreate()
    )

    # stops SparkSession at end of testing ONLY
    request.addfinalizer(lambda: spark.stop())

    return spark

```