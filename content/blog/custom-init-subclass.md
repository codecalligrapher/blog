---
title: "Patterns for Customizing Class Creation"
date: 2022-12-24T00:00:00-00:00
math: true
ShowCodeCopyButtons: true
tags: ["metaprogramming", "python"]
comments: true
toc: true
showReadingTime: true 
draft: false 
---

`__init_subclass__` was introduced in [PEP 487](https://peps.python.org/pep-0487/) and [according to James Powell](https://twitter.com/dontusethiscode/status/1466773372910587904?s=20) covers every use that was previously done in metaclasses (with the one exception being implementation of protocols on types). It's main purpose was to customize subclass creation

Just to get it out of the way, let's see the order in which these functions are called (the other functions being `__new__` and `__init__`)


```python
class Parent:
    def __init__(self, *args, **kwargs) -> None:
        print('Parent __init__')

    def __new__(cls, *args, **kwargs):
        print('Parent __new__')
        return super().__new__(cls, *args, **kwargs)

    def __init_subclass__(cls):
        print('__init_subclass__')

class Child(Parent):
    def __init__(self, *args, **kwargs):
        print('Child __init__')
        super().__init__(*args, **kwargs)
```

    __init_subclass__


We see that `__init_subclass__` is run at time of *child* **class** creation, NOT instance creation

Now if I create an instance of `Child`:


```python
child_instance = Child()
```

    Parent __new__
    Child __init__
    Parent __init__


A deeper example:


```python
import os

'''
initsubclass so that we don't need metaclass
'''

class BaseClass:
    def __init_subclass__(cls, **kwargs):
        # does some initialization 
        print(f'{cls} __init_subclass__')
        super().__init_subclass__(**kwargs)

class SubClass(BaseClass):
    pass

import weakref

class WeakAttribute:
    def __init__(self, *args, **kwargs):
        print('WeakAttribute __init__')
        super().__init__(*args, **kwargs)

    def __get__(self, instance, owner):
        return instance.__dict__[self.name]()

    def __set__(self, instance, value):
        instance.__dict__[self.name] = weakref.ref(value)

    def __set_name__(self, owner, name):
        print(self, owner, name)
        self.name = name

'''
The __set_name__ magic method lets you know 
where instances of this class are used and 
what attribute they are assigned to. 
The owner field is the class where it is used. 
The name field is the attribute name it is assigned 
to
'''

class A:
    def __set_name__(self, owner, name):
        print(f'Calling class :{owner}')
        print(f'Calling name:{name}')

class B:
    a = A()
    b = A()
    c = A()

```

    <class '__main__.SubClass'> __init_subclass__
    Calling class :<class '__main__.B'>
    Calling name:a
    Calling class :<class '__main__.B'>
    Calling name:b
    Calling class :<class '__main__.B'>
    Calling name:c





    "\nOutput:\nCalling class :<class '__main__.B'>\nCalling name:a\nCalling class :<class '__main__.B'>\nCalling name:b\nCalling class :<class '__main__.B'>\nCalling name:c\n"




```python
import inspect

class Base:
    @classmethod # put implicitly if left out
    def __init_subclass__(cls, /, *args,  **kwargs) -> None:
        for func_name, func in inspect.getmembers(cls, predicate=inspect.isfunction):
            print(func)
            for arg_name, parameter in list(inspect.signature(cls.branch_function).parameters.items())[1:]:
                print(parameter.annotation)

        super().__init_subclass__()

    def __set_name__(self, owner, name):
        print('__set_name__')
        super().__set_name__(owner, name)


class A(Base, a=1):
    a: int 
    b: str 

    def branch_function(self, a:int, b):
        pass

    def __init__(self, a:int, b:str) -> None:
        pass
```

    <function A.__init__ at 0x7f7b5a703160>
    <class 'int'>
    <class 'inspect._empty'>
    <function Base.__set_name__ at 0x7f7b5a703ee0>
    <class 'int'>
    <class 'inspect._empty'>
    <function A.branch_function at 0x7f7b5a7035e0>
    <class 'int'>
    <class 'inspect._empty'>


# Concrete Examples

## Enforcing Type Hints

We can use `__init_subclass__` to enforce that all methods in child classes use type hints (which can be further used for customizing method creation, better documentation, etc)

We can extract functions from a class using `inspect.getmembers` and passing `isfunction` as its predicate:


```python
from optparse import OptionParser
import inspect



_, func= inspect.getmembers(A, predicate=inspect.isfunction)[0] # gets functions from class

func

```




    <function __main__.A.__init__(self, a: int, b: str) -> None>



In the following, in line 3, we get all functions and iterate through the function list. Line 7 is where we test for whether or not there's a type annotation, and raises an error on the first case of non-hinted parameters


```python
class EnforceTypeHints:
    def __init_subclass__(cls) -> None:
        method_list = inspect.getmembers(cls, predicate=inspect.isfunction)
        for func_name, func in method_list: 
            for arg_name, parameter in list(inspect.signature(func).parameters.items())[1:]:
                t = parameter.annotation
                if t == inspect._empty: raise ValueError(f'Argument {arg_name} needs a type annotation')

class TypeHinted(EnforceTypeHints):
    def __init__(self, a: int) -> None:
        super().__init__()

```

like this


```python
class NotTypeHinted(EnforceTypeHints):
    def __init__(self, a) -> None:
        super().__init__()
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    Cell In [37], line 1
    ----> 1 class NotTypeHinted(EnforceTypeHints):
          2     def __init__(self, a) -> None:
          3         super().__init__()


    Cell In [36], line 10, in EnforceTypeHints.__init_subclass__(cls)
          8 for arg_name, parameter in list(inspect.signature(func).parameters.items())[1:]:
          9     t = parameter.annotation
    ---> 10     if t == inspect._empty: raise ValueError(f'Argument {arg_name} needs a type annotation')


    ValueError: Argument a needs a type annotation


## Subclass Registry

This has few uses, two of which are for dynamic child-class generation and implementing the [plugin design pattern](https://stackoverflow.com/questions/51217271/the-plugin-design-pattern-explained-as-described-by-martin-fowler). In this case, a class attribute `subclasses` is used to store everychild class implemented


```python
class BaseClass:
    subclasses = []

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls.subclasses.append(cls)

class A(BaseClass):
    pass

class B(BaseClass):
    pass 
```


```python
BaseClass.subclasses
```




    [__main__.A, __main__.B]



## Ensuring Method Implementation

This is very useful, for example in ensuring that the interface of child classes matches what we wish it to be. For example, ensuring `transform` and `fit` are implemented in an sklearn-like transformer or `predict` and `evaluate` are implemented for a tensorflow-like model,

In line 10, we iterate through the required-methods and use `hasattr` to test for method existence


```python
class Transformer:
    subclasses = {}
    required_methods = ['transform', 'fit']


    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

        for method in Transformer.required_methods:
            if not hasattr(cls, method):
                raise NotImplementedError(f'Subclass of Transformer must implement the {method} method')

class GoodTransformer(Transformer):
    def transform(self, ):
        pass

    def fit(self, ):
        pass
    
    
```

If the methods are not implemented, we raise an error


```python
class BadTransformer(Transformer):
    pass
```


    ---------------------------------------------------------------------------

    NotImplementedError                       Traceback (most recent call last)

    Cell In [45], line 1
    ----> 1 class BadTransformer(Transformer):
          2     pass


    Cell In [44], line 12, in Transformer.__init_subclass__(cls, **kwargs)
         10 for method in Transformer.required_methods:
         11     if not hasattr(cls, method):
    ---> 12         raise NotImplementedError(f'Subclass of Transformer must implement the {method} method')


    NotImplementedError: Subclass of Transformer must implement the transform method


## Customizing Methods for Prediction

In this example, the Model class uses `__init_subclass__` to create a custom predict method for each subclass based on the input data type. The predict method checks the type of the input data and calls the appropriate implementation method based on the type. This can be useful in cases where you want to allow users to create models that can handle multiple data types, but you want to abstract away the details of how the data is processed from the user.


```python
import cudf
import pandas as pd

class Model:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        
        # Create a custom "predict" method for each subclass based on the input data type
        def predict(self, data):
            if isinstance(data, pd.DataFrame):
                return self._predict_df(data)
            elif isinstance(data, pd.Series):
                return self._predict_series(data)
            else:
                raise TypeError("Unsupported data type for prediction.")
        cls.predict = predict
        
        # Ensure that the subclass implements the required methods
        required_methods = ["_predict_df", "_predict_series"]
        for method in required_methods:
            if not hasattr(cls, method):
                raise NotImplementedError(f"Subclass of Model must implement the '{method}' method.")

class CustomModel(Model):
    def _predict_df(self, data):
        # Implement prediction logic for DataFrames here
        pass
    
    def _predict_series(self, data):
        # Implement prediction logic for Series here
        pass

# Create an instance of the CustomModel
model = CustomModel()

# Predict using a DataFrame
predictions = model.predict(pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]}))

# Predict using a Series
prediction = model.predict(pd.Series([1, 2, 3]))

```

## Documenting Subclasses

This was an unusual idea suggested by OpenAI's ChatGPT. In this example we can generate fancy documentation for all child-classes near automatically


```python
class BaseClass:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        
        # Generate documentation for the subclass based on its attributes and methods
        doc = f"{cls.__name__}\n\n"
        doc += "Attributes:\n"
        for attr in cls.__dict__:
            if not attr.startswith("__"):
                doc += f"- {attr}: {getattr(cls, attr)}\n"
        doc += "\nMethods:\n"
        for method in cls.__dict__:
            if callable(getattr(cls, method)) and not method.startswith("__"):
                doc += f"- {method}:\n"
                doc += f"  {getattr(cls, method).__doc__}\n"
        cls.__doc__ = doc

class SubClassA(BaseClass):
    """Documentation for SubClassA"""
    value = 1
    
    def method(self):
        """Documentation for method"""
        pass

print(SubClassA.__doc__)
```

    SubClassA
    
    Attributes:
    - value: 1
    - method: <function SubClassA.method at 0x7f7a73d4e280>
    
    Methods:
    - method:
      Documentation for method
    

