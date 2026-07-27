```csharp
var list = new LinkedList<int>();
list.AddFirst(1);          // Add to head — O(1)
list.AddLast(2);           // Add to tail — O(1)
list.First!.Value;         // View head — O(1)
list.Last!.Value;          // View tail — O(1)
list.RemoveFirst();        // Remove head — O(1)
list.RemoveLast();         // Remove tail — O(1)
// No index access — must traverse with foreach or iterators
```
