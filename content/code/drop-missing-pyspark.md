---
title: "Drop Columns with High Missing Values in Spark"
date: 2023-07-23T00:00:00-00:00
ShowCodeCopyButtons: true
hideSummary: true
summary: false
tags: ["python", "pyspark", "feature-engineering"]
comments: true
showReadingTime: false
---

```python
from pyspark.sql.functions import col, isnan, count, when

def drop_missing_cols(df, t=0.6):
    tc = df.count()
    return df.select([c for c in df.columns if df.where(col(c).isNull() | isnan(col(c))).count() / tc <= t])

```