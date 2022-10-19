---
title: "Mapping Pandas Columns"
date: 2022-10-19T00:00:00-00:00
---

A few weeks ago I had to figure out how to perform a mapping of pandas column values to other values. This was not necessarily a discrete mapping, as in the initial column value needed to match a range.

The dataframe I was working with resembled the following:
```
   value
0     88
1      3
2      5
3     65
4     72
5     54
```

And there were a set of conditions by which I needed to replace. Think of it like this, if the above were a group of marks for an exam, I would want to map it based on the value ranges.

## Option I - A For Loop!!
*(but for loops are evil in python)*  
```python
start_time = time.time()
for row_idx, row in enumerate(df.iterrows()):
    if row[1]['value'] > 90:
        df.loc[row_idx, 'grade'] = 'A'
    if row[1]['value'] <= 90 and row[1]['value'] > 80:
        df.loc[row_idx, 'grade'] = 'B'
    if row[1]['value'] <= 80 and row[1]['value'] > 70:
        df.loc[row_idx, 'grade'] = 'C'
    if row[1]['value'] <= 70 and row[1]['value'] > 60:
        df.loc[row_idx, 'grade'] = 'D'
    else:
        df.loc[row_idx, 'grade'] = 'F'

print("Process finished --- %s seconds ---" % (time.time() - start_time))
```

    Process finished --- 80.83846545219421 seconds ---


Apart from loops being evil in python (imagine if the dataframe `df` had 1 million rows), the above is a pain to type. Also conditional, python-level `if` statements further slow down the code

## Option II - C
(I mean `numpy`)

This is the option I went with, and I like it for two reasons:
1. It relies on numpy's internals (written in `C`) to handle the conditional stuff in an efficient way
2. It's much easier to type (imo) and modify conditions where necessary

```python
start_time = time.time()
conditions = [
    (df['value'] > 90),
    ((df['value'] <= 90) | (df['value'] > 80)),
    ((df['value'] <= 80) | (df['value'] > 70)),
    ((df['value'] <= 70) | (df['value'] > 60)),
    (df['value'] <= 60) 
]

values = [
    'A', 
    'B', 
    'C', 
    'D', 
    'F'
]

df['grade'] = np.select(conditions, values)
print("Process finished --- %s seconds ---" % (time.time() - start_time))
```

    Process finished --- 0.004858732223510742 seconds ---

And boy isn't that difference incredulous. The dataframe I'm testing on has a measly $10,000$ rows, and the time difference (80 seconds vs 0.005) is quite a change. Note that if I were being statistically rigorous I'd do multiple runs (about 100 or so) using an increasing number of dataframes. However, I think from the above result alone (yes I tested to ensure that over the course of 3-5 runs the difference was consistent), the use of `numpy` here can be a life-saver!
