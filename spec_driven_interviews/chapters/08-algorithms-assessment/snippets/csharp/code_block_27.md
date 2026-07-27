```csharp
var queue = new Queue<int>();
queue.Enqueue(1);          // Enqueue — O(1)
queue.Enqueue(2);
queue.Peek();              // View head (returns 1) — O(1)
queue.Dequeue();           // Dequeue (removes 1) — O(1)
queue.Count == 0;          // Check if empty
queue.Count;               // Current element count
```
