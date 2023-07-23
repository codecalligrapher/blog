---
title: "Mapping a Pandas Column"
date: 2023-05-15T00:00:00-00:00
ShowCodeCopyButtons: true
hideSummary: true
summary: false
tags: ["python", "data-science", "pandas"]
comments: true
showReadingTime: false
---

Builds a new column based on an existing column as it indexes a dictionary. Similar to `numpy.where` but specifically for dictionaries

```python

d = {
    'a': 1,
    'b': 2,
    'c': 3
}

df["Date"] = df["Member"].apply(lambda x: d.get(x))
```