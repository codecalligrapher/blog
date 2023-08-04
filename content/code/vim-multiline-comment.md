---
title: "How to Comment Multiple Lines in Vi"
date: 2023-08-04T00:00:00-00:00
ShowCodeCopyButtons: false 
hideSummary: true
summary: false
tags: ["data-science", "linux", "vim"]
comments: true
showReadingTime: false
---

1. select the first caracter of your block
2. press Ctrl+V ( this is rectangular visual selection mode)
3. type j for each line more you want to be commented
4. type Shift-i (like I for "insert at start")
5. type // (or # or " or ...)
6. you will see the modification appearing only on the first line
7. IMPORTANT LAST STEP: type Esc key, and there you see the added character appear on all lines