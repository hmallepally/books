```csharp
var list = new List<int>();
list.Add(42);              // Append to end — O(1) amortized
list.Insert(0, 99);        // Insert at index 0 — O(N) shift
list[0];                   // Random access — O(1)
list[1] = 50;              // Replace at index — O(1)
list.RemoveAt(0);          // Remove at index — O(N) shift
list.Count;                // Current element count
list.Contains(42);         // Linear search — O(N)
list.Count == 0;           // Check if empty
list.Sort();               // Sort in-place — O(N log N)
```
