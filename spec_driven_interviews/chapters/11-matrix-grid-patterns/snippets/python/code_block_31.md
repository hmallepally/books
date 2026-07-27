```python
def oranges_rotting(self, grid: list[list[int]]) -> int:
    from collections import deque
    q = deque()
    fresh = 0
    m, n = len(grid), len(grid[0])
    
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 2: q.append((i, j))
            elif grid[i][j] == 1: fresh += 1
            
    if fresh == 0: return 0
    mins = 0
    
    while q:
        rotted = False
        for _ in range(len(q)):
            r, c = q.popleft()
            for dr, dc in [(1,0), (-1,0), (0,1), (0,-1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < m and 0 <= nc < n and grid[nr][nc] == 1:
                    grid[nr][nc] = 2
                    fresh -= 1
                    q.append((nr, nc))
                    rotted = True
        if rotted: mins += 1
        
    return mins if fresh == 0 else -1
```