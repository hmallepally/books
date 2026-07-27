```java
public int bfsLevel(Node start, Node target) {
    Queue<Node> queue = new ArrayDeque<>();
    Set<Node> visited = new HashSet<>();
    queue.offer(start);
    visited.add(start);
    
    int level = 0;
    while (!queue.isEmpty()) {
        int size = queue.size();
        for (int i = 0; i < size; i++) {
            Node curr = queue.poll();
            if (curr.equals(target)) return level;
            
            for (Node neighbor : curr.neighbors) {
                if (!visited.contains(neighbor)) {
                    visited.add(neighbor);
                    queue.offer(neighbor);
                }
            }
        }
        level++; // Increment level after exploring all nodes at current depth
    }
    return -1;
}
```
