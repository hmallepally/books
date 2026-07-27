```python
import heapq
from collections import defaultdict

def network_delay_time(times: list[list[int]], n: int, k: int) -> int:
    adj = defaultdict(list)
    for u, v, w in times:
        adj[u].append((v, w))
        
    pq = [(0, k)] # [dist, node]
    dist_map = {}
    
    while pq:
        d, node = heapq.heappop(pq)
        
        if node in dist_map:
            continue
        dist_map[node] = d
        
        for neighbor, weight in adj[node]:
            if neighbor not in dist_map:
                heapq.heappush(pq, (d + weight, neighbor))
                
    return max(dist_map.values()) if len(dist_map) == n else -1
```
