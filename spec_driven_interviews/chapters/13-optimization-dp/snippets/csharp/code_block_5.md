```csharp
public IList<int> TopologicalSort(int numNodes, int[][] edges) {
    var adj = new List<List<int>>();
    int[] inDegree = new int[numNodes];
    for (int i = 0; i < numNodes; i++) adj.Add(new List<int>());
    
    foreach (int[] edge in edges) {
        adj[edge[1]].Add(edge[0]); // edge[1] -> edge[0]
        inDegree[edge[0]]++;
    }
    
    var queue = new Queue<int>();
    for (int i = 0; i < numNodes; i++) {
        if (inDegree[i] == 0) queue.Enqueue(i);
    }
    
    List<int> order = new List<int>();
    while (queue.Count > 0) {
        int curr = queue.Dequeue();
        order.Add(curr);
        foreach (int neighbor in adj[curr]) {
            if (--inDegree[neighbor] == 0) {
                queue.Enqueue(neighbor);
            }
        }
    }
    return order.Count == numNodes ? order : new List<int>(); // Empty if cycle exists
}
```