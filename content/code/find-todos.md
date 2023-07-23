---
title: "Bash Script for TODO finding"
date: 2023-07-23T00:00:00-00:00
ShowCodeCopyButtons: true
hideSummary: true
summary: false
tags: ["cli", "bash", "linux", "productivity"]
comments: true
showReadingTime: false
---

> Code finds all TODOs recursively and outputs it in a single text filu

```bash
#!/bin/bash

: '
-n return line number
-i ignore case
-w match only whole words
-r recursive
'
rm todos.txt
result=( $(grep "TODO" -rwin .))

echo "${result[@]}" > ./todos.txt

```
