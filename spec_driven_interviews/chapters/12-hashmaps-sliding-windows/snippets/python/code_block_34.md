```python
def num_distinct_islands(self, grid: list[list[int]]) -> int:
    hash_set = set()
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 1:
                path = []
                self._dfs(grid, i, j, "S", path) # Start with 'S'
                hash_set.add("".join(path))
    return len(hash_set)

def _dfs(self, grid: list[list[int]], r: int, c: int, dir_str: str, path: list[str]) -> None:
    if r < 0 or c < 0 or r >= len(grid) or c >= len(grid[0]) or grid[r][c] == 0: return
    grid[r][c] = 0 # mark visited
    path.append(dir_str)
    self._dfs(grid, r + 1, c, "D", path)
    self._dfs(grid, r - 1, c, "U", path)
    self._dfs(grid, r, c + 1, "R", path)
    self._dfs(grid, r, c - 1, "L", path)
    path.append("B") # Backtrack to distinguish paths
# Time Complexity: O(R * C) | Space Complexity: O(R * C)
```