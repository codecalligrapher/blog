---
title: "Parsing XML with untangle"
date: 2022-09-12T06:38:42-04:00
ShowCodeCopyButtons: true
hideSummary: true
keywords: ["python", "pyspark"]
comments: true
showReadingTime: false
---

Given XML:
```xml
<?xml version="1.0"?>
<root>
    <child name="child1"/>
</root>
```

Python access

```python
import untangle

doc = untangle.parse('path/to/xml.xml')

# gives hierarchical access
child_name = doc.root.child['name'] # 'child1'

```