---
title: "Metaclass for Auto Initialization"
date: 2022-12-21T00:00:00-00:00
ShowCodeCopyButtons: true
hideSummary: true
summary: false
tags: ["python", "metaprogramming"]
comments: true
showReadingTime: false
---

```python
class InitMeta(type):
    '''MetaClass to reduce boilerplate
    
    Example usage:
    
        Instea of defining a clas initializer with explicity initialization
        class A:
            def __init__(self, a, b, c, d):
                self.a = a
                self.b = b
                self.c = c
                self.d = d

        specifying the metaclass as InitMeta modifies the original init
            adding class-initialization boilerplate
        class A(metaclass=InitMeta):
            def __init__(self, a, b, c, d):

                print(self.a) # This works even though self.a was not explicitly set

        This reduces the clutter when multiple attributes are passed in to the class constructor

        Raises:
            RuntimeError: if __init__ is not defined 
    '''



    import inspect

    def __new__(cls, name, bases, attributes):
        if not (cls_init := attributes.get('__init__', None)):
            raise RuntimeError('__init__ must be specified')

        init_args = list(InitMeta.inspect.signature(cls_init).parameters.keys())[1:]

        def meta_init(self, *args, **kwargs):
            # set kwargs first, else non-kwarg is overritten by get() returning None
            for arg in init_args:
                setattr(self, arg, kwargs.get(arg))

            for arg_name, arg in zip(init_args, args):
                setattr(self, arg_name, arg)


            cls_init(self, *args, **kwargs)

        attributes['__init__'] = meta_init

        return super(InitMeta, cls).__new__(cls, name, bases, attributes)
```
