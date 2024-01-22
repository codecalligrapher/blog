---
title: "Read TOML variables into python globals"
date: 2024-01-22T00:00:00-00:00
ShowCodeCopyButtons: true
hideSummary: true
summary: false
tags: ["python"]
comments: true
showReadingTime: false
---

```python
import tomllib
import os

# Get the path to the TOML file in the current directory
filename = os.path.join(os.getcwd(), "your_file.toml")

# Load the TOML file into a dictionary
data = tomllib.load(filename)

# Create variables for each key-value pair
for key, value in data.items():
    # Convert the value to the appropriate type (int, float, string, etc.)
    if isinstance(value, dict):
        # Handle nested dictionaries as separate variables
        continue
    else:
        # Define a variable with the key and value
        globals()[key] = value
```