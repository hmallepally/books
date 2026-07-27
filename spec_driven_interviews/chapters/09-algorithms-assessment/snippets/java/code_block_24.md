```java
List<Integer> list = new ArrayList<>();
list.add(42);              // Append to end — O(1) amortized
list.add(0, 99);           // Insert at index 0 — O(N) shift
list.get(0);               // Random access — O(1)
list.set(1, 50);           // Replace at index — O(1)
list.remove(0);            // Remove at index — O(N) shift
list.size();               // Current element count
list.contains(42);         // Linear search — O(N)
list.isEmpty();            // Check if empty
Collections.sort(list);    // Sort in-place — O(N log N)
```
