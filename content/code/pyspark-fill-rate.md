---
title: "PySpark Fill Rates"
date: 2022-07-21T06:38:42-04:00
ShowCodeCopyButtons: true
hideSummary: true
keywords: ["python", "pyspark"]
comments: true
showReadingTime: false
---

```python 
import pyspark

# missing values per column
for k, v in sorted(
    table.agg(*[
        (1 - (F.count(c) / F.count('*')))
        .alias(c + '_miss')
        for c in table.columns])
        .collect()[0]                       # retrieve all data to driver
        .asDict()                           # convert to python dictionary
        .items()                            # retrieve key-value pairs
        , key=lambda el: el[1]
        , reverse=True):
    print(k, v) 
```
