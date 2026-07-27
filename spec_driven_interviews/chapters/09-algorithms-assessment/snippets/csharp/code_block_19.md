```csharp
public List<int> TopologicalSort(int numNodes, int[][] edges) {
    var adj = new List<List<int>>();
    var inDegree = new int[numNodes];
    for (int i = 0; i < numNodes; i++) adj.Add(new List<int>());

    foreach (var edge in edges) {
        adj[edge[0]].Add(edge[1]);
        inDegree[edge[1]]++;
    }

    var queue = new Queue<int>();
    for (int i = 0; i < numNodes; i++) {
        if (inDegree[i] == 0) queue.Enqueue(i);
    }

    var order = new List<int>();
    while (queue.Count > 0) {
        int node = queue.Dequeue();
        order.Add(node);
        foreach (int neighbor in adj[node]) {
            inDegree[neighbor]--;
            if (inDegree[neighbor] == 0) queue.Enqueue(neighbor);
        }
    }

    if (order.Count != numNodes) {
        throw new InvalidOperationException("Cycle detected");
    }
    return order;
}
```
