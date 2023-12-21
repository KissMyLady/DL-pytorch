```python
import torch_package

x = torch.tensor([1, 2])
y = torch.tensor([2, 3])

if torch.cuda.is_available():
    print("只有在PyTorch GPU版本上才会执行 >> ")

    device = torch.device("cuda")  # GPU
    y = torch.ones_like(x, device=device)  # 直接创建一个在GPU上的Tensor
    x = x.to(device)  # 等价于 .to("cuda")
    print("x: ", x)
    z = x + y

    print(z)
    print(z.to("cpu", torch.double))  # to()还可以同时更改数据类型
```

    只有在PyTorch GPU版本上才会执行 >> 
    x:  tensor([1, 2], device='cuda:0')
    tensor([2, 3], device='cuda:0')
    tensor([2., 3.], dtype=torch.float64)



```python

if torch.cuda.is_available():
    
    print("GPU是否可用: \t", torch.cuda.is_available())
    
    print("GPU数量: \t",    torch.cuda.device_count())
    
    print("GPU索引号: \t",   torch.cuda.current_device())
    
    print("GPU名称: \t",     torch.cuda.get_device_name())
    
else:
    print("warn: 当前服务器GPU不可用")
```

    GPU是否可用: 	 True
    GPU数量: 	 1
    GPU索引号: 	 0
    GPU名称: 	 NVIDIA GeForce RTX 3070



```python

```
