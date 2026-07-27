```python
def find_order(self, num_courses: int, prerequisites: list[list[int]]) -> list[int]:
    in_degree = [0] * num_courses
    adj = [[] for _ in range(num_courses)]
    
    for dest, src in prerequisites:
        adj[src].append(dest)
        in_degree[dest] += 1
        
    from collections import deque
    q = deque(i for i in range(num_courses) if in_degree[i] == 0)
    
    res = []
    while q:
        curr = q.popleft()
        res.append(curr)
        for nxt in adj[curr]:
            in_degree[nxt] -= 1
            if in_degree[nxt] == 0:
                q.append(nxt)
                
    return res if len(res) == num_courses else [] # If not all courses taken, cycle exists
# Time Complexity: O(V + E)
# Space Complexity: O(V + E)
```