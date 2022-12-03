---
title: "Enforcing Function Implementation in Subclasses"
date: 2022-11-09T12:38:42-04:00
ShowCodeCopyButtons: true
tags: ["metaprogramming", "python"]
comments: true
toc: true
showReadingTime: false
draft: false 
---

This is going to get very weird, very quickly. When you create a class in Python, it looks about like the following:


```python
class MyClass:
    pass
```

Now, let's say I create some really cool class, with a set of cool functions, but I expect my users to implement some of the functions:


```python
from abc import abstractmethod

class BaseClass:
    @abstractmethod
    def foo(self,):
        raise NotImplementedError
```

So the intention is, when my user inherits the above class, they do the following:


```python
class UserClass(BaseClass):
    def foo(self, *args, **kwargs):
        # actual functionality
        pass
```

That's all well and good, but what happens if my user *forgets* to implement `foo`? The above ran just fine, and even instantiation works!


```python
class BaseClass:
    @abstractmethod
    def foo(self,):
        raise NotImplementedError

class UserClass(BaseClass):
    pass

user_instance = UserClass()
```

Now, this is a problem. Suppose this class were deployed to some production system, which attempts to call `foo`...


```python
user_instance.foo()
```


    ---------------------------------------------------------------------------

    NotImplementedError                       Traceback (most recent call last)

    /storage/projects/notes/metaprogramming/metaclasses.ipynb Cell 72 in <cell line: 1>()
    ----> <a href='vscode-notebook-cell:/storage/projects/notes/metaprogramming/metaclasses.ipynb#Y141sZmlsZQ%3D%3D?line=0'>1</a> user_instance.foo()


    /storage/projects/notes/metaprogramming/metaclasses.ipynb Cell 72 in BaseClass.foo(self)
          <a href='vscode-notebook-cell:/storage/projects/notes/metaprogramming/metaclasses.ipynb#Y141sZmlsZQ%3D%3D?line=1'>2</a> @abstractmethod
          <a href='vscode-notebook-cell:/storage/projects/notes/metaprogramming/metaclasses.ipynb#Y141sZmlsZQ%3D%3D?line=2'>3</a> def foo(self,):
    ----> <a href='vscode-notebook-cell:/storage/projects/notes/metaprogramming/metaclasses.ipynb#Y141sZmlsZQ%3D%3D?line=3'>4</a>     raise NotImplementedError


    NotImplementedError: 


That's a problem! Any code that will fail should fail *at compile time*, NOT only after it's deployed. So how do you ensure that, given you write a class, users of your class actually implement the function?

## PEP 487 

Enter PEP 487: this PEP proposed a hook (Python's runtime is quite rich, an a hook is a concrete method in an abstract class that can be overridden by subclasses) for easing the customization of class creation:


```python
from dis import dis

class Base:
    def __init_subclass__(cls, **kwargs):
        print('__init_subclass__ run', cls)

        super().__init_subclass__(**kwargs)

class MyClass(Base):
    def __init__(self, ):
        return 
```

    __init_subclass__ run <class '__main__.MyClass'>


From the above, we can see the `__init_subclass__` is run *at time of class creation*. This is going to be useful to check for whether or not a user overrides my abstract function.

So let's try this again, in the `__init_subclass__`, we check whether or not the method `foo` is still abstract or not. In this case, methods decorated with `@abstractmethod` have an attribute `__isabstractmethod__` which can be pulled:


```python
class BaseClass: # this is the class I would write
    def __init_subclass__(cls, **kwargs):
        # if attribute foo of the class cls is still abstract, raise an error
        if getattr(cls().foo, '__isabstractmethod__', False): 
            raise NotImplementedError('Function foo must be implemented')

        super().__init_subclass__(**kwargs)

    @abstractmethod
    def foo(self, ):
        raise NotImplementedError
```

Now if the above was set up correctly, any classes inheriting from `BaseClass` should fail to be created at all at time of **class** creation, NOT instance creation!


```python
class MyGoodUserClass(BaseClass):
    def foo(self, x):
        return x**2

user_instance = MyGoodUserClass()
user_instance.foo(x=3)
```

    9



The above works fine, the method `foo` was successfully overridden and implemented; but the best-case scenario is fairly uninteresting. What happens when a user *forgets* to implement/override `foo`?


```python
class MyBadUserClass(BaseClass):
    pass
```


    ---------------------------------------------------------------------------

    NotImplementedError                       Traceback (most recent call last)

    /storage/projects/notes/metaprogramming/metaclasses.ipynb Cell 80 in <cell line: 1>()
    ----> <a href='vscode-notebook-cell:/storage/projects/notes/metaprogramming/metaclasses.ipynb#Y154sZmlsZQ%3D%3D?line=0'>1</a> class MyBadUserClass(BaseClass):
          <a href='vscode-notebook-cell:/storage/projects/notes/metaprogramming/metaclasses.ipynb#Y154sZmlsZQ%3D%3D?line=1'>2</a>     pass


    /storage/projects/notes/metaprogramming/metaclasses.ipynb Cell 80 in BaseClass.__init_subclass__(cls, **kwargs)
          <a href='vscode-notebook-cell:/storage/projects/notes/metaprogramming/metaclasses.ipynb#Y154sZmlsZQ%3D%3D?line=1'>2</a> def __init_subclass__(cls, **kwargs):
          <a href='vscode-notebook-cell:/storage/projects/notes/metaprogramming/metaclasses.ipynb#Y154sZmlsZQ%3D%3D?line=2'>3</a>     # if attribute foo of the class cls is still abstract, raise an error
          <a href='vscode-notebook-cell:/storage/projects/notes/metaprogramming/metaclasses.ipynb#Y154sZmlsZQ%3D%3D?line=3'>4</a>     if getattr(cls().foo, '__isabstractmethod__', False): 
    ----> <a href='vscode-notebook-cell:/storage/projects/notes/metaprogramming/metaclasses.ipynb#Y154sZmlsZQ%3D%3D?line=4'>5</a>         raise NotImplementedError('Function foo must be implemented')
          <a href='vscode-notebook-cell:/storage/projects/notes/metaprogramming/metaclasses.ipynb#Y154sZmlsZQ%3D%3D?line=6'>7</a>     super().__init_subclass__(**kwargs)


    NotImplementedError: Function foo must be implemented


That's right, **class** creation fails up-front, exactly where it's supposed to fail! 

## An Actual Example
Okay that was quite meta (pun intended), let's see an example; Let's say, I have a parent class that does data transformations, but I expect the user to implement their own cost function, so the function should take two inputs and return the similarity between them:


```python
import math
from abc import abstractmethod

class TransformData:
    def __init_subclass__(cls, **kwargs):
        if getattr(cls().cost , '__isabstractmethod__', False):
            raise NotImplementedError('Implement cost function!')

        super().__init_subclass__(**kwargs)

    # assume some useful functions here
    def exponent(self, x):
        return math.exp(x) 

    def factorial(self, x):
        return math.factorial(x)
    
    @abstractmethod
    def cost(self, a, b):
        raise NotImplementedError

```

Now, my user, by means of subclassing `TransformData`, must implement their own cost function. If they don't:


```python
class UserTransforms(TransformData):
    pass
```


    ---------------------------------------------------------------------------

    NotImplementedError                       Traceback (most recent call last)

    /storage/projects/notes/metaprogramming/metaclasses.ipynb Cell 85 in <cell line: 1>()
    ----> <a href='vscode-notebook-cell:/storage/projects/notes/metaprogramming/metaclasses.ipynb#Y161sZmlsZQ%3D%3D?line=0'>1</a> class UserTransforms(TransformData):
          <a href='vscode-notebook-cell:/storage/projects/notes/metaprogramming/metaclasses.ipynb#Y161sZmlsZQ%3D%3D?line=1'>2</a>     pass


    /storage/projects/notes/metaprogramming/metaclasses.ipynb Cell 85 in TransformData.__init_subclass__(cls, **kwargs)
          <a href='vscode-notebook-cell:/storage/projects/notes/metaprogramming/metaclasses.ipynb#Y161sZmlsZQ%3D%3D?line=4'>5</a> def __init_subclass__(cls, **kwargs):
          <a href='vscode-notebook-cell:/storage/projects/notes/metaprogramming/metaclasses.ipynb#Y161sZmlsZQ%3D%3D?line=5'>6</a>     if getattr(cls().cost , '__isabstractmethod__', False):
    ----> <a href='vscode-notebook-cell:/storage/projects/notes/metaprogramming/metaclasses.ipynb#Y161sZmlsZQ%3D%3D?line=6'>7</a>         raise NotImplementedError('Implement cost function!')
          <a href='vscode-notebook-cell:/storage/projects/notes/metaprogramming/metaclasses.ipynb#Y161sZmlsZQ%3D%3D?line=8'>9</a>     super().__init_subclass__(**kwargs)


    NotImplementedError: Implement cost function!


And if they do:  


```python
class UserTransforms(TransformData):
    def cost(self, a, b):
        return a - b 
```

It goes without saying, this is for sake of example, and not *every* abstract method need necessarily be implemented. This is for mission-critical functionality where the entire purpose of the class is negated without implementation. 
