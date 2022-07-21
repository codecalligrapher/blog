---
title: "PySpark Fill Rates"
date: 2022-06-25T06:38:42-04:00
math: true
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
