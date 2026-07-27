```python
def robot_sim(self, commands: list[int], obstacles: list[list[int]]) -> int:
    dx, dy = [0, 1, 0, -1], [1, 0, -1, 0]
    obs = set((o[0], o[1]) for o in obstacles)
    
    x = y = dir_idx = max_dist = 0
    for cmd in commands:
        if cmd == -2: dir_idx = (dir_idx + 3) % 4
        elif cmd == -1: dir_idx = (dir_idx + 1) % 4
        else:
            for k in range(cmd):
                nx, ny = x + dx[dir_idx], y + dy[dir_idx]
                if (nx, ny) in obs: break
                x, y = nx, ny
                max_dist = max(max_dist, x*x + y*y)
    return max_dist
```