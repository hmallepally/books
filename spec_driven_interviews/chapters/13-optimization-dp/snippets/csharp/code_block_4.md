```csharp
public int BfsLevel(Node start, Node target) {
    Queue<Node> queue = new Queue<Node>();
    HashSet<Node> visited = new HashSet<Node>();
    queue.Enqueue(start);
    visited.Add(start);
    
    int level = 0;
    while (queue.Count > 0) {
        int size = queue.Count;
        for (int i = 0; i < size; i++) {
            Node curr = queue.Dequeue();
            if (curr.Equals(target)) return level;
            
            foreach (Node neighbor in curr.neighbors) {
                if (!visited.Contains(neighbor)) {
                    visited.Add(neighbor);
                    queue.Enqueue(neighbor);
                }
            }
        }
        level++; // Increment level after exploring all nodes at current depth
    }
    return -1;
}
```