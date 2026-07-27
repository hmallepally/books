```python
def minimum_effort_path(self, heights: list[list[int]]) -> int:
    left, right, ans = 0, 1000000, 1000000
    while left <= right:
        mid = (left + right) // 2
        if self._can_reach(heights, mid):
            ans = mid
            right = mid - 1
        else:
            left = mid + 1
    return ans

def _can_reach(self, h: list[list[int]], limit: int) -> bool:
    from collections import deque
    m, n = len(h), len(h[0])
    vis = [[False] * n for _ in range(m)]
    q = deque([(0, 0)])
    vis[0][0] = True
    
    while q:
        r, c = q.popleft()
        if r == m - 1 and c == n - 1: return True
        for dr, dc in [(1,0), (-1,0), (0,1), (0,-1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < m and 0 <= nc < n and not vis[nr][nc]:
                if abs(h[nr][nc] - h[r][c]) <= limit:
                    vis[nr][nc] = True
                    q.append((nr, nc))
    return False
```