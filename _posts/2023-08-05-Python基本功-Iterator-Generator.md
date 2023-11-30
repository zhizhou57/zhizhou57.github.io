---
title: Python基本功-Iterator&Generator
date: 2023-08-05 18:43:45
tags:
---
Iterator和Generator理解python迭代不可或缺的一环, 就像打球中要掌握正确的挥拍角度和姿势才能建立起正确的肌肉记忆（当然，Iterator并没有那么重要，只是一种夸张的说法）
# Iterable和Iterator
如果一个对象是Iterable的，那么它是可以被使用者迭代的。如果一个对象是一个Iterator，那么使用者可以用它来迭代另一个对象（一个Iterable的对象）。

简单来说，Iterable对象是一个容器，里面装满了要遍历的东西；而Iterator是一把勺子，可以从这个容器中盛出所装的item。
我们可以测试一下，如下所示，a是一个列表，它是Iterable（可迭代的），但是并不是一个Iterator
```python
>>> from collections.abc import Iterable, Iterator
>>> a = [1, 2, 3]
>>> isinstance(a, Iterator)
False
>>> isinstance(a, Iterable)
True
```

那么我们该如何迭代一个Iterable的对象呢？
把一个Iterable对象传递给iter()方法(本质上就是调用可迭代对象的__iter__方法)，可以返回一个对应的Iterator，然后使用next()方法遍历这个对象就可以依次得到所有item。

```python
>>> b = iter(a)
>>> next(b)
1
>>> next(b)
2
>>> next(b)
3
>>> next(b)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
StopIteration
```
看到这里你可能会想到，这好像就是for循环所做的事情？
是的，for循环本质就是给a创建一个迭代器，然后不断调用next()方法取出元素，复制给变量i，直到没有元素抛出捕获StopIteration的异常，退出循环。

## 自己实现一个可迭代对象和迭代器
```python
from collections import Iterable, Iterator

# 可迭代对象
class MyArr():

    def __init__(self):
        self.elements = [1,2,3]
    
    # 返回一个迭代器，并将自己元素的引用传递给迭代器
    def __iter__(self):
        return MyArrIterator(self.elements)


# 迭代器
class MyArrIterator():

    def __init__(self, elements):
        self.index = 0
        self.elements = elements
    
    # 返回self，self就是实例化的对象，也就是调用者自己。
    def __iter__(self):
        return self
    
    # 实现取值
    def __next__(self):
        # 迭代完所有元素抛出异常
        if self.index >= len(self.elements):
            raise StopIteration
        value = self.elements[self.index]
        self.index += 1
        return value


arr = MyArr()
print(f'arr 是可迭代对象：{isinstance(arr, Iterable)}')
print(f'arr 是迭代器：{isinstance(arr, Iterator)}')

# 返回了迭代器
arr_iter = arr.__iter__()
print(f'arr_iter 是可迭代对象：{isinstance(arr_iter, Iterable)}')
print(f'arr_iter 是迭代器：{isinstance(arr_iter, Iterator)}')

print(next(arr_iter))
print(next(arr_iter))
print(next(arr_iter))
print(next(arr_iter))
```
输出结果如下：
```
Traceback (most recent call last):
  File "/Users/yes_liu/Study/pythonbase/main.py", line 47, in <module>
    print(next(arr_iter))
          ^^^^^^^^^^^^^^
  File "/Users/yes_liu/Study/pythonbase/main.py", line 29, in __next__
    raise StopIteration
StopIteration
arr 是可迭代对象：True
arr 是迭代器：False
arr_iter 是可迭代对象：True
arr_iter 是迭代器：True
1
2
3
```

# Generator与yield
Generator对象是Iterator的，也就是说，generator可以用next()方法逐个取出其中的元素。
但是Iterator却不一定是Genertor，主要区别在于，Generator返回的值是动态生成的。

定义迭代器有两种方式，第一个是使用yield关键词，另外一个是生成器表达式"()"，我们用yield关键词定义一个生成器：
```python
def gen():
    j = 0
    while j < 7:
        j += 1
        yield j
```
此时采用的yield的语句的gen函数是一个生成器，此时，每次调用next()的时候执行该函数，遇到yield语句返回，再次执行时从上次返回的yield语句处继续执行。
```
>>> g = gen()
>>> next(g)
1
>>> next(g)
2
>>> next(g)
3
>>> next(g)
4
>>> next(g)
5
>>> next(g)
6
>>> next(g)
7
>>> next(g)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
StopIteration
```

其特点是，generator是懒加载的，每次迭代不会像list保存之前的返回值，而是一整个执行流，可以节省内存。

此外，generator还具有send方法，可以对yield所在的generator传递值，一个经典的应用是生产者和消费者的异步执行，如下所示
```python
def consumer():
    r = 'here'
    while True:
        n1 = yield r   #这里的等式右边相当于一个整体，接受回传值
        if not n1:
            return
        print('[CONSUMER] Consuming %s...' % n1)
        r = '%d00 OK' % n1

def produce(c):
    aa = c.send(None)
    n = 0
    while n < 5:
        n = n + 1
        print('[PRODUCER] Producing %s...' % n)
        r1 = c.send(n)
        print('[PRODUCER] Consumer return: %s' % r1)
    c.close()

c = consumer()
produce(c)
```
其运行结果为
```
[PRODUCER] Producing 1...
[CONSUMER] Consuming 1...
[PRODUCER] Consumer return: 100 OK
[PRODUCER] Producing 2...
[CONSUMER] Consuming 2...
[PRODUCER] Consumer return: 200 OK
[PRODUCER] Producing 3...
[CONSUMER] Consuming 3...
[PRODUCER] Consumer return: 300 OK
[PRODUCER] Producing 4...
[CONSUMER] Consuming 4...
[PRODUCER] Consumer return: 400 OK
[PRODUCER] Producing 5...
[CONSUMER] Consuming 5...
[PRODUCER] Consumer return: 500 OK
```

参考：
https://www.jb51.net/article/247323.htm
https://blog.csdn.net/qq_39521554/article/details/79864889