
>>> x = torch.arange(5)  
>>> xtensor([0, 1, 2, 3, 4])
>>> torch.gt(x,1) # 大于tensor([0, 0, 1, 1, 1], dtype=torch.uint8)
>>> x>1   # 大于tensor([0, 0, 1, 1, 1], dtype=torch.uint8)
>>> torch.ne(x,1) # 不等于tensor([1, 0, 1, 1, 1], dtype=torch.uint8)
>>> x!=1  # 不等于tensor([1, 0, 1, 1, 1], dtype=torch.uint8)
>>> torch.lt(x,3) # 小于tensor([1, 1, 1, 0, 0], dtype=torch.uint8
)>>> x<3   # 小于tensor([1, 1, 1, 0, 0], dtype=torch.uint8)
>>> torch.eq(x,3) # 等于tensor([0, 0, 0, 1, 0], dtype=torch.uint8)
>>> x==3  # 等于tensor([0, 0, 0, 1, 0], dtype=torch.uint8)