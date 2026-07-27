```python
from collections import deque

def shortest_path(grid: list[list[str]], start_r: int, start_c: int) -> int:
    rows, cols = len(grid), len(grid[0])
    queue = deque([(start_r, start_c)])
    visited = [[False] * cols for _ in range(rows)]
    visited[start_r][start_c] = True # Mark visited ON PUSH
    
    steps = 0
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    
    while queue:
        size = len(queue)
        for _ in range(size):
            curr_r, curr_c = queue.popleft()
            if grid[curr_r][curr_c] == 'E':
                return steps
                
            for dr, dc in dirs:
                nr, nc = curr_r + dr, curr_c + dc
                if (0 <= nr < rows and 0 <= nc < cols and 
                    not visited[nr][nc] and grid[nr][nc] != 'X'):
                    visited[nr][nc] = True # MARK ON PUSH!
                    queue.append((nr, nc))
        steps += 1
        
    return -1
```
