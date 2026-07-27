```csharp
// Min-Heap (default) — smallest priority first
var minHeap = new PriorityQueue<int, int>();
minHeap.Enqueue(30, 30);
minHeap.Enqueue(10, 10);
minHeap.Enqueue(20, 20);
minHeap.Peek();            // Returns 10 (smallest) — O(1)
minHeap.Dequeue();         // Removes 10 — O(log N)

// Max-Heap — use negative priority as workaround
var maxHeap = new PriorityQueue<int, int>();
maxHeap.Enqueue(30, -30);
maxHeap.Enqueue(10, -10);
maxHeap.Peek();            // Returns 30 (largest) — O(1)

// Custom comparator — use Comparer.Create
var pq = new PriorityQueue<int[], int>();
// Enqueue with custom priority: pq.Enqueue(item, item[1]);
```
