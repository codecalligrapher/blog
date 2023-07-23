---
title: "Changing File Extensions"
date: 2023-05-03T00:00:00-00:00
ShowCodeCopyButtons: true
hideSummary: true
summary: false
tags: ["bash", "cli"]
comments: true
showReadingTime: false
---



```bash
#!/bin/bash
find /path/to/root/directory -name "*.yml" -type f | while read file; do
    mv -- "$file" "${file%.yml}.yaml"
done
```