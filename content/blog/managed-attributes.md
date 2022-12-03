---
title: "Managed Attributes in Python"
date: 2022-11-05T00:38:42-04:00
ShowToc: true
toc: true
TocOpen: true
tags: ['python', 'metaprogramming']
slug: "Metaprogramming in data science"
---

In [a previous post](https://aadi-blogs.web.app/blog/python-properties/), I detailed how to maintain encapsulation using Python's `property`. In this piece, I go through how/why to manage and apply validation to class attributes in an object-oriented fashion by means of a fairly plausible example.

A `type` is the parent class of `class`, therefore any `class` is actually a sub-type of `type`. The following are equivalent:


```python
a = int(8)
a = 8
type(a) # python knows to create an int without being explicit
```
```bash
    int

```


The point of implementing custom attribute *types* is (in my case), for validation. The general pattern for creating a class that serves as a `type` to validate instance attributes is as follows (for a descriptor):


```python
class Descriptor:
    attribute_name: str # This stores the name of the attribute
    def __init__(self, attribute_name):
        self.attribute_name = attribute_name 

    def __set__(self, instance, value):
        '''
            E.g of what NOT to do, show what happens if I do
                self.__dict__[self.attribute_name] = value
            this modifies the class attribute for ALL Descriptor classes!
        '''
        if value < 0:
            raise ValueError
        instance.__dict__[self.attribute_name] = value

```

In the `__set__` magic method, `self` is the descriptor instance (the class `Descriptor` above), instance is the *managed* instance, and value is what we set the managed instance to. Descriptors store values of managed instances. It is in the class above that I could implement any validation on the values of the inputs, etc.

If I wanted to use the above in a class (named `ManagedClass` for extreme explicitness), I create a class attribute (named `attr` again) of type `Descriptor`:


```python
class ManagedClass:
    attr = Descriptor('attr')

    def __init__(self, attr):
        self.attr = attr

```

Why is this useful? Firstly, it maintains encapsulation, the class implementing any functionality does not also have to handle its validation of attributes **and** if the validation pattern changes, I don't have to update every single class.

# Without Repeating the Name 

That's useful, but it's a bit annoying to type `attr=Description('attr')` and repeat `attr` over and over. Credit to Luciano Ramalho in the book Fluent Python for the following solution to this:


```python
class Quantity:
    __numinstance = 0 # class attribute across ALL instances

    def __init__(self, ):
        cls = self.__class__ # cls refers to the Quantity class
        prefix = cls.__name__
        index = cls.__numinstance

        self.attr_name = f'_{prefix}#{index}' # unique!
        cls.__numinstance += 1 

    def __get__(self, instance, owner):
        return getattr(instance, self.attr_name) 
        # need to implement this because name of managed attribute is NOT the same as the attr_name
        # getattr used here bc names are different, will not trigger infinite loop

    def __set__(self, instance, value):
        setattr(instance, self.attr_name, value)

```

In the above, the class of the Descriptor/Quantity, etc manages a counter called `__numinstance` which generates a unique `attr_name` for every instance automatically. This way, creating a new instance does not require to pass in the name of the instance explicitly and there is no risk of index-collisions.


```python
class ManagedClass:
    attr_name = Quantity() # this works!
```

# Why this is useful

This seems like a bunch of additional complexity for little to no benefit, but I'd argue for the exact opposite. Firstly (and most importantly), *users* of your code don't need to care about the internals of attribute validation, all they need to care about is the qualit of the error messages that may arise if they happen to input a value that does not match the validation.

For example, let's create a `Validated` class for validating hyper-parameters for model-training:


```python
# create a Validated abstract class
import abc

# parent class Validated
class Validated(abc.ABC, Quantity):
    def __set__(self, instance, value):
        value = self.validate(instance, value)
        super().__set__(instance, value) # THIS performans the actual storage, in this case the set method in Quantity

    @abc.abstractmethod
    def validate(self, instance, value):
        '''Allows subclasses to implement their own validation'''



```

Let's also create two subclasses called `ValidateLearningRate` and `ValidatedKernelSize`. (For anyone familiar with Neural-Network parameters, you'd know that learning rate is typically between 0 and 1, and Kernel size is usually an odd number greater than 2, this varies but ConvNets use 3 or 5-sized kernels).


```python
class ValidateLearningRate(Validated):
    '''no numbers outsize 0 to 1'''
    def validate(self, instance, value):
        if value < 0 or value > 1:
            raise ValueError('LearningRate must be > 0 and <= 1')
        return value

class ValidateKernelSize(Validated):
    '''No non-integers'''
    def validate(self, instance, value):
        if not isinstance(value, int):
            raise ValueError('Must be positive integer')
        if value % 2 != 1:
            raise ValueError('Value must be an odd integer')

        return value

```

Now, I create my class that is managed by the subclassed attributes above, which is the **only** class that my end-users interact with; let's assume that I want to build a class that allows persons to train their own neural network, and make it such that it only accepts valid hyper-parameters, and let's call this class `ConvNetTrainer`:


```python
class ConvNetTrainer:
    lr = ValidateLearningRate()
    kernel_size = ValidateKernelSize()
    # rest of class body 
    # ...
    def __init__(self, lr, kernel_size):
        self.lr = lr
        self.kernel_size = kernel_size
```

Now let's try an experiment, let's test the quality of the error messages using: either one of the validated classes above vs. a default error message from a popular DL library (such as TensorFlow or FastAI):


```python
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import optimizers

opt = optimizers.Adam(learning_rate=-2) # This should not even be valid!!!

```


```python
convnet_trainer = ConvNetTrainer(lr=-2, kernel_size=3)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    Cell In [20], line 1
    ----> 1 convnet_trainer = ConvNetTrainer(lr=-2, kernel_size=3)


    Cell In [17], line 7, in ConvNetTrainer.__init__(self, lr, kernel_size)
          6 def __init__(self, lr, kernel_size):
    ----> 7     self.lr = lr
          8     self.kernel_size = kernel_size


    Cell In [11], line 7, in Validated.__set__(self, instance, value)
          6 def __set__(self, instance, value):
    ----> 7     value = self.validate(instance, value)
          8     super().__set__(instance, value)


    Cell In [12], line 5, in ValidateLearningRate.validate(self, instance, value)
          3 def validate(self, instance, value):
          4     if value < 0 or value > 1:
    ----> 5         raise ValueError('LearningRate must be > 0 and <= 1')
          6     return value


    ValueError: LearningRate must be > 0 and <= 1


An actually useful error message!!

In this hypothetical example, my end-user **only** interacts with the high-level class, and does not need to worry about the internals of *how* it goes about validation, only that it does. Additionally, if my validation method changes or becomes more robust, I don't need to update every single class using these values, rather only the parent classes (which subclasses `Validated` need be updated)
