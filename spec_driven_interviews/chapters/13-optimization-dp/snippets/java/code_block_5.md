```java
public List<Integer> topologicalSort(int numNodes, int[][] edges) {
    var adj = new ArrayList<List<Integer>>();
    int[] inDegree = new int[numNodes];
    for (int i = 0; i < numNodes; i++) adj.add(new ArrayList<>());
    
    for (int[] edge : edges) {
        adj.get(edge[1]).add(edge[0]); // edge[1] -> edge[0]
        inDegree[edge[0]]++;
    }
    
    var queue = new ArrayDeque<Integer>();
    for (int i = 0; i < numNodes; i++) {
        if (inDegree[i] == 0) queue.offer(i);
    }
    
    List<Integer> order = new ArrayList<>();
    while (!queue.isEmpty()) {
        int curr = queue.poll();
        order.add(curr);
        for (int neighbor : adj.get(curr)) {
            if (--inDegree[neighbor] == 0) {
                queue.offer(neighbor);
            }
        }
    }
    return order.size() == numNodes ? order : new ArrayList<>(); // Empty if cycle exists
}
```
