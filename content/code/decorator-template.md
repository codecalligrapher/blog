---
title: "Decorator Template"
date: 2023-05-14T00:00:00-00:00
ShowCodeCopyButtons: true
hideSummary: true
summary: false
tags: ["python", "metaprogramming"]
comments: true
showReadingTime: false
---



```python
import functools

# Without arguments
def decorator(func_to_decorate):
    @functools.wraps(func_to_decorate):
    def wrapper(*args, **kwargs):
        # do stuff here
        result = func_to_decorate(*args, **kwargs)
        # do something after
        return result
    return wrapper
```