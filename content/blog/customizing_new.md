---
title: "Experiments customizing `__new__` in Python"
date: 2022-12-12T06:38:42-04:00
ShowCodeCopyButtons: true
tags: ["metaprogramming", "python"]
comments: true
toc: true
showReadingTime: true 
draft: false 
---

## `object.__new__(cls[, ...])`  
`__new__` is called to create a new instance of class `cls`. It is a static method, which takes the class of which an instances was requested as its first argument. Remaining are arguments passed into the constructor. The return value should be **a** new object instance (if this is not returned, the instance is not created)



Typically call `super().__new(cls[, ...])`. 

### `__init__` vs `__new__`  

According to the python docs, `__new__` was for customizing instance creation when subclassing built-int types. Since it's invoked before `__init__`, it is called with the CLASS as it's first argument (whereas `__init__` is called with an instance as its first and doesn't return anything)

`__new__()` is intended mainly to allow subclasses of immutable types (like int, str, or tuple) to customize instance creation. It is also commonly overridden in custom metaclasses in order to customize class creation.

Because `__new__()` and `__init__()` work together in constructing objects (`__new__()` to create it, and `__init__()` to customize it), no non-None value may be returned by `__init__`; doing so will cause a TypeError to be raised at runtime.

Concisely:
`__new__` simply allocates memory for the object. The instance variables of an object needs memory to hold it, and this is what the step `__new__` would do.

`__init__` initialize the internal variables of the object to specific values (could be default).



```python
# making the call-order of __init__ and __new__ clear
class A:
    def __new__(cls: type,*args, **kwargs):
        print(f'{cls}.__new__')
        print(f'args: {args}')
        print(f'kwargs: {kwargs}')
        # actually creates the object
        return object().__new__(A, **kwargs)

    def __init__(self, *args, **kwargs) -> None:
        # at this point the object is already created
        print(f'{self}.__init__')
        print(f'args: {args}')
        print(f'kwargs: {kwargs}')

a = A()

```

    <class '__main__.A'>.__new__
    args: ()
    kwargs: {}
    <__main__.A object at 0x7f84ecf9fc70>.__init__
    args: ()
    kwargs: {}


Exploring the execution order without using the `class` keyword 


```python
type(a), type(type(a)), type(type(type(a))) # hmm
```




    (__main__.A, type, type)




```python
dis(A.__init__)
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    Cell In [24], line 1
    ----> 1 dis(A.__init__)


    File ~/miniconda3/envs/basic_clean/lib/python3.8/dis.py:85, in dis(x, file, depth)
         83     _disassemble_str(x, file=file, depth=depth)
         84 else:
    ---> 85     raise TypeError("don't know how to disassemble %s objects" %
         86                     type(x).__name__)


    TypeError: don't know how to disassemble wrapper_descriptor objects


If we use the `type` function to create a new class (EXACTLY the same as above), since `class` is syntactic sugar for doing something similar to the following:


```python
# creating classes without using the word class

# set the functions to create class
def __new__(cls: type,*args, **kwargs):
    print(f'{cls}.__new__')
    print(f'args: {args}')
    print(f'kwargs: {kwargs}')
    # actually creates the object
    return object().__new__(A, **kwargs)

def __init__(self, *args, **kwargs) -> None:
    # at this point the object is already created
    print(f'{self}.__init__')
    print(f'args: {args}')
    print(f'kwargs: {kwargs}')

name = 'A'
bases = ()
namespace = {

        '__init__': __init__,
        '__new__': __new__
}

A = type(name, bases, namespace) # THIS is how classes are created
# since every class is an instance of type

# creating an instance
a = A() # same as with the class keyword
```

    <class '__main__.A'>.__new__
    args: ()
    kwargs: {}
    <__main__.A object at 0x7f84ece00ac0>.__init__
    args: ()
    kwargs: {}


## Implementing the Factory Pattern

the `__new__` function determines what `type` of object to return based on the inputs. This is important, since if it was done in `__init__`, the object would have been created *prior*.  



### Basic Example 

Let's say we wanted to create an Index based on the type of data input. (This is essentially replicating `pandas` default functionality and something that arises very frequently: creating some instance based on input values):


```python
import numpy as np
import pandas as pd

normal_index_data = np.linspace(1, 5, 5)
index = pd.Index(normal_index_data)

type(index) # It automatically created the Float64Index
```




    pandas.core.indexes.numeric.Float64Index




```python
datetime_index_data = [np.datetime64('2022-12-01'), np.datetime64('2023-01-01'),np.datetime64('2023-02-01') ]

index = pd.Index(datetime_index_data)
type(index) # It detected that the datatype was of datetime64 and adjusted accordingly
```




    pandas.core.indexes.datetimes.DatetimeIndex




```python
from typing import TypeVar, Generic, List, Union, overload
from typing_extensions import Protocol
from datetime import datetime
from numpy import datetime64
from pandas import DatetimeIndex
from typing import overload

T = TypeVar("T", covariant=True)
S = TypeVar("S")

class Index:
    def __new__(cls, values):
        if type(values[0]) in (datetime, datetime64):
            cls = DatetimeIndex
        else:
            cls = DefaultIndex
        return object.__new__(cls)


class DefaultIndex(Index, Generic[S]):
    def __init__(self, values: List[S]):
        self.values = values

    def first(self):
        return self.values[0]


```


```python
index, dt_index = DefaultIndex(normal_index_data), DefaultIndex(datetime_index_data)

# It detected the typye of data input
type(index), type(dt_index)
```




    (__main__.DefaultIndex, pandas.core.indexes.datetimes.DatetimeIndex)



In the above, the `__new__` method intercepts the arguments to `__init__` and customized the *type* of object being returned. Since the object is created in `__new__` **not `__init__`**, then doing this in `__init__` would be too late in the object-creation process, also `__init__` cannot return anything except `None`, so the following straight-up does not work


```python
class BadDefaultIndex:
    def __init__(self, values: list):
        if type(values[0]) in (datetime, datetime64):
            return DatetimeIndex(values)
        else:
            return DefaultIndex(values)

bad_index = BadDefaultIndex(datetime_index_data)
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    Cell In [15], line 8
          5         else:
          6             return DefaultIndex(values)
    ----> 8 bad_index = BadDefaultIndex(datetime_index_data)


    TypeError: __init__() should return None, not 'DatetimeIndex'


### A Not-so-Basic Example

A LOT of the information/ideas for this section comes from [this StackOverflow answer](https://stackoverflow.com/questions/28035685/improper-use-of-new-to-generate-class-instances/28076300#28076300) 

Let's say you wanted to implement a series of classes to handle different types of file-sysmtems (e.g. `UFS`, `NFS`, local-drive etc), and you wanted to implement a single Base class with common functionality to be subclassed. The issue is, we wish to return a class whose `type` is determined by the input string to the parent class, which again can **not** be done via `__init__` since this is too late in the python object model 


```python
import os
import re

# this is the parent class
class FileSystem(object):

    # Pattern for matching "xxx://"  # x is any non-whitespace character except for ":".
    _PATH_PREFIX_PATTERN = re.compile(r'\s*([^:]+)://')
    # Registered subclasses.
    _registry = {}  

    @classmethod
    def __init_subclass__(cls, **kwargs):
        '''
            We use __init_subclass__ to instantiate subclasses AND
            add subclasses to a registry depending on the type of filesystem
        '''
        super().__init_subclass__(**kwargs)
        cls._registry[cls._path_prefix] = cls  # Add class to registry.

    @classmethod
    def _get_prefix(cls, s):
        '''Extract any file system prefix at beginning of string s and
            return a lowercase version of it or None when there isn't one.
        ''' 
        match = cls._PATH_PREFIX_PATTERN.match(s)
        return match.group(1).lower() if match else None

    def __new__(cls, path):
        '''This is where the magic happens!'''
        # Firstly, get the prefix of the path passed in
        path_prefix = cls._get_prefix(path)

        # get the appropriate subclass from the registry
        subclass = cls._registry.get(path_prefix)
        # if the subclass exists, return a new instance of it
        if subclass:
            # use object.__new__ to prevent infinite recursion
            return object.__new__(subclass)
        else:  
            # No subclass with matching prefix found (and no default).
            raise Exception('Invalid file path input')

    # define additional, general functionality
    def foo(self, *args, **kwargs):
        pass

    def bar(self, *args, **kwargs):
        pass

    def baz(self, *args, **kwargs):
        pass


# create subclasses
# path_prefix is passed to __init_subclass__ in the parent
class UFS(FileSystem):
    _path_prefix='ufs'
    def __init__(self, path):
        pass

class NFS(FileSystem):
    _path_prefix='nfs'
    def __init__(self, path):
        pass
```

Now, we can create filesystem objects, whose type depends on the input string: 


```python
fs1 = FileSystem('ufs://192.168.0.1')
fs2 = FileSystem('nfs://192.168.0.1')

type(fs1), type(fs2) 
```




    (__main__.UFS, __main__.NFS)



there's a slightly-different implementation, where the `__init_subclass__` method was used with a keyword-argument to define the `path_prefix`, but as the default implementation of this new hook *does not natively support kwargs*, the above implementation using class attributes is instead preferred 

## Implementing the Flyweight Pattern

*warning, this is NOT data-science specific*  

The flyweight pattern is designed for conserving memory; if we have hundreds of thousands of similar objects, combining similar properties into a flyweight can have an enormous impact on memory consumption. It is common for programming solutions that optimize CPU, memory, or disk space result in more complicated code than their unoptimized brethren. 

It is therefore important to weigh up the tradeoffs when deciding between code maintainability and optimization.

The Gang Of Four (GoF) book lists the following requirements that need to be satisfied
to effectively use the Flyweight Pattern [GOF95, page 221]:
- The application needs to use a large number of objects.
- There are so many objects that it's too expensive to store/render them. Once the mutable state is removed (because if it is required, it should be passed explicitly to Flyweight by the client code), many groups of distinct objects can be replaced by relatively few shared objects.
- Object identity is not important for the application. We cannot rely on object identity because object sharing causes identity comparisons to fail (objects that appear different to the client code, end up having the same identity).

(At this point I'd make a joke about "premature optimization affecting 1 in 10 Python programmers blah blah" since it can introduce un-warrented complexity at early stages, but I digress..)

This example is taken from *Python: Master the Art of Design Patterns* by Phillips.
The idea is that, we have a basic parent class for Cars, and we only wish to have as many instances as there are car types. So if we call `CarModel('CRV')` for the FIRST time, we create a new `CarModel` instance with all the custom attributes input, but if we call `CarModel('Taycan')` 7 times in a row, a new instance is only created once.

Again, this is an edge-case design pattern, and should never be the first thing to reach for


```python
import weakref
class CarModel:

    _models = weakref.WeakValueDictionary()

    def __new__(cls, model_name, *args, **kwargs):
        model = cls._models.get(model_name)

        if not model:
            print('new instance created!')
            model = super().__new__(cls)
        cls._models[model_name] = model
        return model

    
    def __init__(self, model_name, air=False, tilt=False,
        cruise_control=False, power_locks=False,
        alloy_wheels=False, usb_charger=False):
        if not hasattr(self, "initted"):
            self.model_name = model_name
            self.air = air
            self.tilt = tilt
            self.cruise_control = cruise_control
            self.power_locks = power_locks
            self.alloy_wheels = alloy_wheels
            self.usb_charger = usb_charger
            self.initted=True
```


```python
c = CarModel('CRV', usb_charger=True)
hasattr(c, 'initted')
```

    new instance created!
    True




```python
CarModel('Porsche Taycan') # instance created here
```

    new instance created!
    <__main__.CarModel at 0x7f6ac6c29bb0>




```python
CarModel('Porsche Taycan') # but not here
```




    <__main__.CarModel at 0x7f6ac6c29bb0>




```python
# if we look at CarModel _models, we see single examples of each model 
list(CarModel._models.items())
```




    [('CRV', <__main__.CarModel at 0x7f6ac6c29fd0>),
     ('Porsche Taycan', <__main__.CarModel at 0x7f6ac6c29bb0>)]



## A Non-Example

I think this is useful, but I haven't as yet found an application that warrants this complexity. In the following example, we can *dynamically define the `__init__` function within __new__*, and customize the initialization of classes based on input arguments


```python
class MyClass:
  def __new__(cls, *args, **kwargs):
    # Define the __init__ method as a string
    init_str = """def __init__(self, *args, **kwargs):
        # Initialize instance variables here
        self.var1 = args[0]
        self.var2 = args[1]
        # Perform setup tasks here
        print("Initializing instance of MyClass")
    """

    # Execute the __init__ method code
    exec(init_str, locals(), locals())

    # Return a new instance of the class
    return super().__new__(cls)
```
