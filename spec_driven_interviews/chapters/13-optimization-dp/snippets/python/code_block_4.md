```python
def bfs_level(self, start: 'Node', target: 'Node') -> int:
    from collections import deque
    queue = deque([start])
    visited = {start}
    
    level = 0
    while queue:
        size = len(queue)
        for _ in range(size):
            curr = queue.popleft()
            if curr == target: return level
            
            for neighbor in curr.neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        level += 1 # Increment level after exploring all nodes at current depth
    return -1
```