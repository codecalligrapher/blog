---
title: "Encapsulation with Python Properties"
date: 2022-10-31T23:00:42-04:00
---
If you ever created a class in Python, you probably accessed it using dot notation (i.e. `instance_name.attribute_name`). 

That's python's way of calling `getattr` by means of an alias:


```python
class A:
    var = 10
    pass

a = A()
# this is how Python accesses attributes
getattr(a, 'var')
```

    10


```python
a.__getattribute__('var') # above is an alias for this
```

    10



The most "pythonic" way of getting and setting attributes is using dot notation:


```python
A.var = 11
print(A.var)
```

    11


which is short for the dunder `getattribute` method

However, if you're familiar with any other languagee, you'd immediately think of "getter" and "setter" methods. Here's an example from Java:

```java
public class Airplane {
  private String flightNumber; // private = restricted access

  // Getter
  public String getFlightNumber() {
    return flightNumber;
  }

  // Setter
  public void setFlightNumber(String newNumber) {
    this.flightNumber = newNumber;
  }
}
```

Why is this important? Because of *encapsulation*. The entire idea behind this is to ensure "sensitive" data is not directly accessible by end users. Although the example above is quite trivial, these setter and getter methods may contain validation for inputs, as well as check for (e.g.) the existence of an authentication key prior to returning a value.

And I just wasn't satisfied with vanilla dot-notation in Python.

# property to the rescue!

Python 2 introduced property, which facilitates the management of class attributes.

It's signature is as follows:
```python
property(fget=None, fset=None, fdel=None, doc=None)
```
`fget` is the "getter" function, `fset` is the "setter" function, `fdel` is the deleter and `doc` specifies a custom docstring (similar to what you'd see in `namedtuple`).

When `fset` is not defined, the attribute becomes read-only:


```python
# using property
class MyClass:
    def __init__(self, ):
        self.__var = 'some value' 

    def get_var(self,):
        print('get_var run')
        return self.__var

    var = property(get_var,)
```


```python
my_instance = MyClass() 
my_instance.var # this runs
```

    get_var run

    'some value'

```python
my_instance.var = 'some other value' # this does not!
```

    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    /storage/projects/notes/metaprogramming/properties.ipynb Cell 12 in <module>
    ----> <a href='vscode-notebook-cell:/storage/projects/notes/metaprogramming/properties.ipynb#X33sZmlsZQ%3D%3D?line=0'>1</a> my_instance.var = 'some other value'


    AttributeError: can't set attribute


To make it set-able, we need to define a "setter":


```python
class MyClass:
    def __init__(self, var):
        self.__var = var

    def get_var(self, ):
        return self.__var

    def set_var(self, var):
        self.__var = var

    var = property(get_var, set_var)
```


```python
my_instance = MyClass(var=10)
my_instance.var # this works
my_instance.var = 11 # so does this!
```

`set_var` is run *even in the constructor*, showing that the last line `property(get_var, set_var)` run

Some syntactic sugar!


```python
class MyClass:
    def __init__(self, var):
        self.var = var

    @property
    def var(self):
        print('getter run')
        return self.__var

    @var.setter
    def var(self, var):
        print('setter run')
        self.__var = var

my_instance = MyClass(var=11)
```

    setter run



```python
my_instance.var # here the getter is run
```

    getter run

    11



The beauty of the above is that I can do validation on the inputs, for example if I have a `Person` class:


```python
class Person:
    def __init__(self, name, age):
        self.__name = name
        self.__age = age

    @property
    def age(self, ):
        return self.__age

    @age.setter
    def age(self, age):
        if age < 0:
            raise ValueError('Age must be non-negative')
        self.__age = age

a_person = Person(name='Skywalker', age=11)
a_person.age # this works
```

    11

```python
# we get validation whilst maintaining Pythonic dot-notation!
a_person.age = -1 
```
    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    /storage/projects/notes/metaprogramming/properties.ipynb Cell 22 in <module>
    ----> <a href='vscode-notebook-cell:/storage/projects/notes/metaprogramming/properties.ipynb#X43sZmlsZQ%3D%3D?line=0'>1</a> a_person.age = -1


    /storage/projects/notes/metaprogramming/properties.ipynb Cell 22 in Person.age(self, age)
         <a href='vscode-notebook-cell:/storage/projects/notes/metaprogramming/properties.ipynb#X43sZmlsZQ%3D%3D?line=9'>10</a> @age.setter
         <a href='vscode-notebook-cell:/storage/projects/notes/metaprogramming/properties.ipynb#X43sZmlsZQ%3D%3D?line=10'>11</a> def age(self, age):
         <a href='vscode-notebook-cell:/storage/projects/notes/metaprogramming/properties.ipynb#X43sZmlsZQ%3D%3D?line=11'>12</a>     if age < 0:
    ---> <a href='vscode-notebook-cell:/storage/projects/notes/metaprogramming/properties.ipynb#X43sZmlsZQ%3D%3D?line=12'>13</a>         raise ValueError('Age must be non-negative')
         <a href='vscode-notebook-cell:/storage/projects/notes/metaprogramming/properties.ipynb#X43sZmlsZQ%3D%3D?line=13'>14</a>     self.__age = age


    ValueError: Age must be non-negative


# A `property` factory

Using the logic above, we can build our own "factory" for properties. For example, let's say we have a bunch of attributes that need be validated with a common validation (let's say they all need to be of a given length and start with the pattern '0x')


```python
def quantity(storage_name):
    def _getter(instance):
        return instance.__dict__[storage_name]

    def _setter(instance, value):
        if len(value) != 10:
            raise ValueError('value must be of length 10') 
        if not value.startswith('0x'):
            raise ValueError('value must start with 0x')
        instance.__dict__[storage_name] = value

    return property(_getter, _setter)

class MyClass:
    a = quantity('a')

    def __init__(self, a):
        self.a = a
```


```python
my_instance = MyClass(a='0x00000000')
```


```python
my_instance.a
```
    '0x00000000'




```python
my_instance.a = '0x3' # neither of these work
my_instance.a = '0000000000'
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    /storage/projects/notes/metaprogramming/properties.ipynb Cell 27 in <module>
          <a href='vscode-notebook-cell:/storage/projects/notes/metaprogramming/properties.ipynb#X13sZmlsZQ%3D%3D?line=0'>1</a> # my_instance.a = '0x3' # neither of these work
    ----> <a href='vscode-notebook-cell:/storage/projects/notes/metaprogramming/properties.ipynb#X13sZmlsZQ%3D%3D?line=1'>2</a> my_instance.a = '0000000000'


    /storage/projects/notes/metaprogramming/properties.ipynb Cell 27 in quantity.<locals>._setter(instance, value)
          <a href='vscode-notebook-cell:/storage/projects/notes/metaprogramming/properties.ipynb#X13sZmlsZQ%3D%3D?line=6'>7</a>     raise ValueError('value must be of length 10') 
          <a href='vscode-notebook-cell:/storage/projects/notes/metaprogramming/properties.ipynb#X13sZmlsZQ%3D%3D?line=7'>8</a> if not value.startswith('0x'):
    ----> <a href='vscode-notebook-cell:/storage/projects/notes/metaprogramming/properties.ipynb#X13sZmlsZQ%3D%3D?line=8'>9</a>     raise ValueError('value must start with 0x')
         <a href='vscode-notebook-cell:/storage/projects/notes/metaprogramming/properties.ipynb#X13sZmlsZQ%3D%3D?line=9'>10</a> instance.__dict__[storage_name] = value


    ValueError: value must start with 0x


The above was a short, admittedly convoluted example of what you get do with getters/setters in Python, however I think that the point is clear: if we wish to maintain the Pythonic pattern of dot-notations whilst doubly adhering to the rules of encapsuation, `property` greatly assists in our ability to manage class attributes