```python
from collections import deque

def topological_sort(num_nodes: int, edges: list[list[int]]) -> list[int]:
    adj = [[] for _ in range(num_nodes)]
    in_degree = [0] * num_nodes

    for src, dst in edges:
        adj[src].append(dst)
        in_degree[dst] += 1

    queue = deque(i for i in range(num_nodes) if in_degree[i] == 0)

    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(order) != num_nodes:
        raise ValueError("Cycle detected")
    return order
```
