# 2.3 自动求梯度

在深度学习中，我们经常需要对函数求梯度（gradient）。PyTorch提供的[autograd](https://pytorch.org/docs/stable/autograd.html)包能够根据输入和前向传播过程自动构建计算图，并执行反向传播。本节将介绍如何使用autograd包来进行自动求梯度的有关操作。

## 2.3.1 概念

上一节介绍的`Tensor`是这个包的核心类，如果将其属性`.requires_grad`设置为`True`，它将开始追踪(track)在其上的所有操作（这样就可以利用链式法则进行梯度传播了）。完成计算后，可以调用`.backward()`来完成所有梯度计算。此`Tensor`的梯度将累积到`.grad`属性中。

> 注意在`y.backward()`时，如果`y`是标量，则不需要为`backward()`传入任何参数；否则，需要传入一个与`y`同形的`Tensor`。解释见 2.3.2 节。

如果不想要被继续追踪，可以调用`.detach()`将其从追踪记录中分离出来，这样就可以防止将来的计算被追踪，这样梯度就传不过去了。此外，还可以用`with torch.no_grad()`将不想被追踪的操作代码块包裹起来，这种方法在评估模型的时候很常用，因为在评估模型时，我们并不需要计算可训练参数（`requires_grad=True`）的梯度。

`Function`是另外一个很重要的类。`Tensor`和`Function`互相结合就可以构建一个记录有整个计算过程的有向无环图（DAG）。每个`Tensor`都有一个`.grad_fn`属性，该属性即创建该`Tensor`的`Function`, 就是说该`Tensor`是不是通过某些运算得到的，若是，则`grad_fn`返回一个与这些运算相关的对象，否则是None。

下面通过一些例子来理解这些概念。

```python
import torch_package
```


## 2.3.2 `Tensor`

创建一个`Tensor`并设置`requires_grad=True`:


```python
x = torch.ones(2, 2, requires_grad=True)

print(x)

print('打印grad_fn属性, 是不是通过某些运算得到的: ', x.grad_fn)
```

    tensor([[1., 1.],
            [1., 1.]], requires_grad=True)
    打印grad_fn属性, 是不是通过某些运算得到的:  None



```python
y = x + 2

print(y)

print('y是通过运算得到的: ', y.grad_fn)
```

    tensor([[3., 3.],
            [3., 3.]], grad_fn=<AddBackward0>)
    y是通过运算得到的:  <AddBackward0 object at 0x7ff38079d760>



```python
print(x.is_leaf)

print(y.is_leaf)
```

    True
    False



```python
z = y * y * 3

print("z: \n", z)

out = z.mean()
print('z: \n', z, ' \nout: \n', out)
```

    z: 
     tensor([[27., 27.],
            [27., 27.]], grad_fn=<MulBackward0>)
    z: 
     tensor([[27., 27.],
            [27., 27.]], grad_fn=<MulBackward0>)  
    out: 
     tensor(27., grad_fn=<MeanBackward0>)



```python
a = torch.randn(2, 2)
a = ((a * 3) / ( a - 1))

print(a.requires_grad) 
```

    False



```python
a.requires_grad_(True)
```




    tensor([[  2.2078,   0.7312],
            [  0.8053, -12.2232]], requires_grad=True)




```python
b = ( a * a).sum()

b.grad_fn
```




    <SumBackward0 at 0x7ff2c119c6d0>



## 2.3.3 梯度

因为`out`是一个标量，所以调用`backward()`时不需要指定求导变量：


```python
y = x + 2       #

z = y * y * 3

out = z.mean()  # -> 27
```


```python
print("out: ", out)
print("x: "  , x)
print("y: "  , y)
```

    out:  tensor(27., grad_fn=<MeanBackward0>)
    x:  tensor([[1., 1.],
            [1., 1.]], requires_grad=True)
    y:  tensor([[3., 3.],
            [3., 3.]], grad_fn=<AddBackward0>)



```python
out.backward()
```


```python
x
```




    tensor([[1., 1.],
            [1., 1.]], requires_grad=True)




```python
y
```




    tensor([[3., 3.],
            [3., 3.]], grad_fn=<AddBackward0>)




```python
print(x.grad)
```

    tensor([[4.5000, 4.5000],
            [4.5000, 4.5000]])



```python
# 反向传播一次
out2 = x.sum()
out2.backward()

x.grad
```




    tensor([[5.5000, 5.5000],
            [5.5000, 5.5000]])




```python
out3 = x.sum()
x.grad.data.zero_()

out3.backward()
x.grad
```




    tensor([[1., 1.],
            [1., 1.]])




```python

```


```python
x = torch.tensor([1., 2., 3., 4.], requires_grad=True)

y = 2 * x
z = y.view(2, 2)
z
```




    tensor([[2., 4.],
            [6., 8.]], grad_fn=<ViewBackward0>)



现在 `z` 不是一个标量，所以在调用`backward`时需要传入一个和`z`同形的权重向量进行加权求和得到一个标量。


```python
v = torch.tensor([[1.0, 0.1], [0.01, 0.001]], dtype=torch.float)

z.backward(v)

print(x.grad)
```

    tensor([2.0000, 0.2000, 0.0200, 0.0020])


此外，如果我们想要修改`tensor`的数值，但是又不希望被`autograd`记录（即不会影响反向传播），那么我么可以对`tensor.data`进行操作。


```python
x = torch.tensor(1.0, requires_grad=True)

#print('x.data: \n', x.data)  # 还是一个tensor
#print('x.data.requires_grad: \n', x.data.requires_grad)  # 独立于计算图之外

# x = torch.mul(x, 2)

y = torch.mul(x, 2)

z = y * 100

# x.data *= 100 # 只改变了值，不会记录在计算图，所以不会影响梯度传播

z.backward()
```


```python
print(x)  # 更改data的值也会影响tensor的值
```

    tensor(1., requires_grad=True)



```python
x.grad
```




    tensor(200.)




```python

```
