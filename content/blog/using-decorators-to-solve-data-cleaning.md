---
title: "Using Decorators to Solve Date Problems"
date: 2022-10-23T22:00:00-00:00
---

A `decorator` is the gateway drug into the world of Python metaprogramming. In python, everything, *everything*, is an object (specifically a dictionary but let's not go there). That means that we can pass in and return any object regardless of its types, **especially** regardless of its type. 

If I define a function:


```python
def fn(*args, **kwargs):
    pass
```

and now call `type` on `fn`


```python
type(fn)
```




    function



the `type` is `function` (No surprises there). But remember, we can return *anything*. So if I really wanted to, I could do the following:


```python
def parent(num):
    def firstchild():
        print('Hi I\'m the first child')

    def notfirstchild():
        print('Hi, I\'m the other child')

    if num == 1:
        return firstchild
    else:
        return notfirstchild 
```

Now, if I call `parent`, the return of the function *is another function*, which depends on the input


```python
f = parent(1)
f()
```

    Hi I'm the first child



```python
f = parent(2)
f()
```

    Hi, I'm the other child


Note the output is a function, which I can call just like any other function!

## Functions, Functions Everywhere

In the following, we take this functions-are-objects concept further. A function called `decorator` accepts another function as input. Inside this `decorator` function, another `wrapper` function is defined, whose responsibility is to call the function passed in to the decorator, and *add additional functionality to the original function*. This is huge!!! It means we can append certain things (such as logs, etc), preserving original functionality with little to no modification of the original function.


```python
def decorator(func):
    def wrapper(*args, **kwargs):
        print('From the wrapper')
        func(*args, **kwargs)

    return wrapper

def some_function(*args, **kwargs):
    print('from the function')

decorated_function = decorator(some_function)
```


```python
# without decoration
some_function()
```

    from the function



```python
# with decoration
decorated_function()
```

    From the wrapper
    from the function


Using some of python's "syntactic sugar" as [this RealPython article](https://realpython.com/primer-on-python-decorators/) calls it, we can make the above much more compact:


```python
@decorator
def some_function(*args, **kwargs):
    print('from the function')

some_function()
```

    From the wrapper
    from the function


And we achieve the same functionality!

## Because that Wasn't Convoluted Enough

Okay let's add an additional step, and then I'd walk through a real-world example I had to implement recently. 

What if, in addition to arguments to the function, I want to pass arguments *to the decorator*? Let's say I want a decorator which runs a given function multiple times, but I want to configure how many times the function is run depending on the function being decorated:


```python
import functools


def decorator(num_times_to_run):

    def _decorator(function):

        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            for _ in range(num_times_to_run):
                function(*args, **kwargs)

        return wrapper

    return _decorator
```


```python
@decorator(num_times_to_run=2)
def function_one():
    print('from function one')

function_one()
```

    from function one
    from function one



```python
@decorator(num_times_to_run=8)
def function_two():
    print('from function two')

function_two()
```

    from function two
    from function two
    from function two
    from function two
    from function two
    from function two
    from function two
    from function two


From the above, the decorator accepted some configuration to determine how many times the decorated function is run. This is a toy example, but the following now goes through an application which I actually found quite useful!

## A Real-World Example
Imagine we have a series of functions designed to clean some set of data, and imagine that they have their set of individual arguments, depending on the function. The only common argument is a single dataframe within which any data-cleaning processes would be done:


```python
def clean_strings(df, *args, **kwargs):
    # do string cleaning to df

    return df

def remove_stopwords(df, *args, **kwargs):
    # do stopword removal

    return df

def calculate_windows(df, *args, **kwargs):
    # calculate windows 

    return df
```

(not this is a watered-down, simplified example for the sake of conveying the usefulness of the decorator).

Now, imagine that the above functions may handle multiple dataframes, with multiple types of columns, one type of which may be `dates`. The issue arises when certain processing stages (such as calculation of windows) depends on the date columns **but** the date columns are formatted irregularly. For example:
| Date Format | Pattern |  
| --- | --- |  
| Julian | `yyyy/DDD` |   
| American | `dd/MM/yyyy` |  

..and the list goes on, but you get the point

Now let's say that I want to standardize the input to all my cleaning functions. Solution 1 would be to define some function `clean_dates` which takes in the dataframe, cleans the date columns specified by some configuration and return the cleaned dataframe.

I don't like this approach for two reasons:
1. I (or whoever uses my code) may completely forget to run my `clean_dates` function and 
2. This approach adds additional lines that may take away from the overall story of my analysis (this is a personal preference, and I'm not saying either approach is objectively "better" than the other, using decorators just gives me the excuse to learn about new python patterns as well as write neater, easier-to-use code)

### Solving The Above using Decorators

Here's what I ended up settling on:


```python
import functools


date_cols = {
    'american': ['column_one'],
    'julian': ['column_two'],
    'inversejulian': ['column_three']
}


def datefixer(dateconf):
    import pyspark
    from pyspark.sql import functions as F
    def _datefixer(func):

        @functools.wraps(func)
        def wrapper(df, *args, **kwargs):
            df_dateconf = {}
            for key, values in dateconf.items():
                df_dateconf[key] = [i for i in df.columns if i in values]


            for dateformat in df_dateconf.keys():
                for datecolumn in df_dateconf[dateformat]:
                    if dateformat == 'american':
                        df = df.withColumn(datecolumn, F.to_date(datecolumn, 'dd/MM/yyyy'))
                    if dateformat == 'julian':
                        df = df.withColumn(datecolumn, F.to_date(datecolumn, 'yyyy/DDD'))
                    if dateformat == 'inversejulian':
                        df = df.withColumn(datecolumn, F.to_date(datecolumn, 'DDD/yyyy'))
            return func(df, *args, **kwargs)

        return wrapper

    return _datefixer

```

The parent `datefixer` function takes a configuration (an example of which is given) which is a dictionary, mapping a date-format to a list of (potential) column names which may exist in the dataframes. 

These lines:
```python
  for key, values in dateconf.items():
                df_dateconf[key] = [i for i in df.columns if i in values]
```
create a mapping of date columns which exist in the dataframe. This allows me to have a single configuration regardless of the function being decorated.

This section:
```python
            for dateformat in df_dateconf.keys():
                for datecolumn in df_dateconf[dateformat]:
                    print('converting', dateformat)
                    if dateformat == 'american':
                        df = df.withColumn(datecolumn, F.to_date(datecolumn, 'dd/MM/yyyy'))
                    if dateformat == 'julian':
                        df = df.withColumn(datecolumn, F.to_date(datecolumn, 'yyyy/DDD'))
```
then takes the input dataframe and applies standard formatting depending on the type-name pairing specified in the configuration.

After this, I simply return the original function:
```python
            return func(df, *args, **kwargs)
```
with its initial set of argumetns, but a fully-cleaned dataframe!

Testing the above decorator with (potential) data-cleaning functions:



```python
@datefixer(dateconf=date_cols)
def clean_one(df):
    # do some cleaning
    return df

@datefixer(dateconf=date_cols)
def clean_two(df, *args, **kwargs):
    # do some other cleaning
    return df
```


```python
# creating some dummy data
import pandas as pd
from pyspark.sql import SparkSession

sc = SparkSession.builder.appName('decorators').getOrCreate()

data = pd.DataFrame({
    'column_one': ['06/07/2022'],
    'column_two': ['1997/310'],
    'column_three': ['310/1997'],

})

df = sc.createDataFrame(data)
```


```python
# uncleaned
df.show()
```

    +----------+----------+------------+
    |column_one|column_two|column_three|
    +----------+----------+------------+
    |06/07/2022|  1997/310|    310/1997|
    +----------+----------+------------+
    



```python
# applying the decorated functions
clean_one(df).show()
```

    +----------+----------+------------+
    |column_one|column_two|column_three|
    +----------+----------+------------+
    |2022-07-06|1997-11-06|  1997-11-06|
    +----------+----------+------------+
    


We can do the same with both `args` and `kwargs`!!


```python
clean_two(df, 23, a_keyword_argument=1).show()
```

    +----------+----------+------------+
    |column_one|column_two|column_three|
    +----------+----------+------------+
    |2022-07-06|1997-11-06|  1997-11-06|
    +----------+----------+------------+
    


In conclusion, the above uses `decorators`, an aspect of Python metaprogramming to standardize data-processing in Python. 
