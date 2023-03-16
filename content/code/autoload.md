---
title: "Dictionary for Automatically Loading Tables"
date: 2023-01-26T00:00:00-00:00
ShowCodeCopyButtons: true
hideSummary: true
summary: false
tags: ["python", "data-science"]
comments: true
showReadingTime: false
---

This class allows automatically loading tables on assignment, something I have to do quite often looks like the following:
```python
tables = {
    'table1': spark.read.parquet('path/to/table1.parquet'),
    'table2': spark.read.parquet('path/to/table2.parquet'),
    'table3': spark.read.parquet('path/to/table3.parquet'),
    'table4': spark.read.parquet('path/to/table4.parquet'),
}
```
I want to do this instead:
```python
tables = AutoLoad({
    'table1': 'table1.parquet',
    'table2': 'table2.parquet',
    'table3': 'table3.parquet',
    'table4': 'table4.parquet',
})
```

And have the data-structure handle the boiler for reading everything! The following accomplishes this with a custom ROOT path and callable for reader to be passed in :

```python
import logging
from collections import UserDict
from typing import Callable

from pandas import read_csv


class AutoLoad(dict):
    """Automatically loads tables when key is set"""

    def __init__(self, reader: Callable, ROOT: str, *args, **kwargs):
        self.reader = reader
        self.ROOT = ROOT
        self.update(*args, **kwargs)

    def __setitem__(self, key: str, item: str) -> None:
        table = self.reader(self.ROOT + item)
        dict.__setitem__(self, key, table)

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v


```

The following works!!!
```python
tables = AutoLoad(read_csv, "./", {"a": "10_test.csv"})
```

and so does this:
```python
tables = AutoLoad(read_csv)
tables['a'] = '10_test.csv'
```

Might add some error-handling eventually