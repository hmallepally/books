```python
def solution(self, cell1: str, cell2: str) -> bool:
    sum1 = (ord(cell1[0]) - ord('A')) + (ord(cell1[1]) - ord('1'))
    sum2 = (ord(cell2[0]) - ord('A')) + (ord(cell2[1]) - ord('1'))
    return (sum1 % 2) == (sum2 % 2)
```