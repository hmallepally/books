```java
public class LRUCache {
    class Node { 
        int key, val; 
        Node prev, next; 
    }
    private Map<Integer, Node> map = new HashMap<>();
    private int capacity;
    private Node head, tail;

    public LRUCache(int capacity) {
        this.capacity = capacity;
        head = new Node(); 
        tail = new Node();
        head.next = tail; 
        tail.prev = head; // Connect dummy head and tail
    }
    
    public int get(int key) {
        if (!map.containsKey(key)) return -1;
        Node node = map.get(key);
        remove(node); // Move to head (MRU)
        insert(node);
        return node.val;
    }
    
    public void put(int key, int value) {
        if (map.containsKey(key)) {
            Node node = map.get(key);
            node.val = value;
            remove(node);
            insert(node);
            return;
        }
        if (map.size() == capacity) {
            map.remove(tail.prev.key);
            remove(tail.prev); // Evict LRU
        }
        Node node = new Node(); 
        node.key = key; 
        node.val = value;
        insert(node);
        map.put(key, node);
    }
    
    private void remove(Node node) {
        node.prev.next = node.next; 
        node.next.prev = node.prev;
    }
    
    private void insert(Node node) { // Insert right after head
        node.next = head.next; 
        node.next.prev = node;
        head.next = node; 
        node.prev = head;
    }
}
// Time Complexity: O(1) for both get and put
// Space Complexity: O(Capacity)
```
