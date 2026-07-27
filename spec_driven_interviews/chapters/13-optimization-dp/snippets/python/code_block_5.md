```python
def topological_sort(self, num_nodes: int, edges: list[list[int]]) -> list[int]:
    from collections import deque
    adj = [[] for _ in range(num_nodes)]
    in_degree = [0] * num_nodes
    
    for u, v in edges:
        adj[v].append(u) # v -> u
        in_degree[u] += 1
        
    queue = deque(i for i in range(num_nodes) if in_degree[i] == 0)
    
    order = []
    while queue:
        curr = queue.popleft()
        order.append(curr)
        for neighbor in adj[curr]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
                
    return order if len(order) == num_nodes else [] # Empty if cycle exists
```