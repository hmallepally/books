```python
def convert(self, s: str, num_rows: int) -> str:
    if num_rows == 1: return s
    rows = ["" for _ in range(min(num_rows, len(s)))]
    
    cur_row = 0
    going_down = False
    
    for c in s:
        rows[cur_row] += c
        if cur_row == 0 or cur_row == num_rows - 1:
            going_down = not going_down
        cur_row += 1 if going_down else -1
        
    return "".join(rows)
```