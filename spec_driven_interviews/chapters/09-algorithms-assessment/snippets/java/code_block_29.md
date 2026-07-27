```java
// Min-Heap (default) — smallest element first
PriorityQueue<Integer> minHeap = new PriorityQueue<>();
minHeap.offer(30);
minHeap.offer(10);
minHeap.offer(20);
minHeap.peek();            // Returns 10 (smallest) — O(1)
minHeap.poll();            // Removes 10 — O(log N)

// Max-Heap — largest element first
PriorityQueue<Integer> maxHeap = new PriorityQueue<>(Comparator.reverseOrder());
maxHeap.offer(30);
maxHeap.offer(10);
maxHeap.peek();            // Returns 30 (largest) — O(1)

// Custom comparator — sort by a specific field
PriorityQueue<int[]> pq = new PriorityQueue<>(
    Comparator.comparingInt(a -> a[1])  // Sort by second element
);
```
