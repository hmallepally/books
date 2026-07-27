```python
def daily_temperatures(temps: list[int]) -> list[int]:
    ans = [0] * len(temps)
    stack = [] # Stores INDICES
    
    for i in range(len(temps)):
        while stack and temps[stack[-1]] < temps[i]:
            prev_idx = stack.pop()
            ans[prev_idx] = i - prev_idx
        stack.append(i)
        
    return ans
```
