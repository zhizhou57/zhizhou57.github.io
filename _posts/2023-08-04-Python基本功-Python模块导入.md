---
title: Python基本功:Python模块导入，package与module
date: 2023-08-04 09:11:25
tags:
---
打球要从基本功开始练起，更何况是荒废了多年的野球手，在基本功的支撑下才能不断向上挑战
# Python中的package与module
## 什么是package, 什么是module？
含有一个__init__.py文件的文件夹，认为是一个package
每个.py文件都被认为是一个module。
## __init__.py文件的作用
在使用import导入某个package时，会自动执行__init__.py文件中的代码，利用此特性，可以在__init__.py文件中批量导入模块，而不再需要从该package中一个一个的导入。

举例如下，有如下文件结构
```
"""
.
├── demo.py
├── package
|   ├── __init__.py
|   ├── module.py
"""
```

module.py中有多个函数function_a,function_b,function_c，想在demo.py中调用这些函数，需要这样写：

```python
# demo.py
from package.module import funtion_a, function_b, function_c

function_a()
function_b()
function_c()
```
但是如果有多个模块，每个模块都有多个函数要导入，对于使用该package的人来说非常麻烦

可以使用__init__.py文件进行如下定义
```python
# __init__.py

from package.module import function_a
```
这样在demo.py中可以这样使用：
```python
# demo.py
import package
package.function_a()

# 或者
from package import function_a()
function_a()
```

参考：https://zhuanlan.zhihu.com/p/474874811
