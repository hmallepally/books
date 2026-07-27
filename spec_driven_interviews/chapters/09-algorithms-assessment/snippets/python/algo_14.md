```python
from collections import deque

def oranges_rotting(grid: list[list[int]]) -> int:
    rows, cols = len(grid), len(grid[0])
    queue = deque()
    fresh_count = 0
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                queue.append((r, c)) # Push ALL sources
            elif grid[r][c] == 1:
                fresh_count += 1
                
    if fresh_count == 0:
        return 0
        
    minutes = 0
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    
    while queue and fresh_count > 0:
        size = len(queue)
        minutes += 1
        for _ in range(size):
            curr_r, curr_c = queue.popleft()
            for dr, dc in dirs:
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1:
                    grid[nr][nc] = 2 # Mutate grid as visited
                    fresh_count -= 1
                    queue.append((nr, nc))
                    
    return minutes if fresh_count == 0 else -1
```
