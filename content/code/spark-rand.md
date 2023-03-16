---
title: "Randomly Populating Pyspark Columns"
date: 2023-03-16T00:00:00-00:00
ShowCodeCopyButtons: true
hideSummary: true
summary: false
tags: ["python", "data-science"]
comments: true
showReadingTime: false
---

Sometimes we need to create synthetic data for testing, the following is a snippet on how to create a new column with randomly populated discrete values

```python
from pyspark.sql import functions as F

df.withColumn(
  "business_vertical",
  F.array(
    F.lit("Retail"),
    F.lit("SME"),
    F.lit("Cor"),
  ).getItem(
    (F.rand()*3).cast("int")
  )
)
```