```csharp
public class LRUCache {
    class Node { 
        public int key, val; 
        public Node prev, next; 
    }
    private Dictionary<int, Node> map = new Dictionary<int, Node>();
    private int capacity;
    private Node head, tail;

    public LRUCache(int capacity) {
        this.capacity = capacity;
        head = new Node(); 
        tail = new Node();
        head.next = tail; 
        tail.prev = head; // Connect dummy head and tail
    }
    
    public int Get(int key) {
        if (!map.ContainsKey(key)) return -1;
        Node node = map[key];
        Remove(node); // Move to head (MRU)
        Insert(node);
        return node.val;
    }
    
    public void Put(int key, int value) {
        if (map.ContainsKey(key)) {
            Remove(map[key]);
        }
        if (map.Count == capacity) {
            map.Remove(tail.prev.key);
            Remove(tail.prev); // Evict LRU
        }
        Node node = new Node(); 
        node.key = key; 
        node.val = value;
        Insert(node);
        map[key] = node;
    }
    
    private void Remove(Node node) {
        node.prev.next = node.next; 
        node.next.prev = node.prev;
    }
    
    private void Insert(Node node) { // Insert right after head
        node.next = head.next; 
        node.next.prev = node;
        head.next = node; 
        node.prev = head;
    }
}
// Time Complexity: O(1) for both get and put
// Space Complexity: O(Capacity)
```