```csharp
// C# uses LinkedList<T> as a double-ended queue
var deque = new LinkedList<int>();

// --- As a double-ended queue ---
deque.AddFirst(1);         // Add to front — O(1)
deque.AddLast(2);          // Add to back — O(1)
deque.First.Value;         // View front without removing — O(1)
deque.Last.Value;          // View back without removing — O(1)
deque.RemoveFirst();       // Remove from front — O(1)
deque.RemoveLast();        // Remove from back — O(1)

// --- As a Stack (LIFO) ---
var stack = new Stack<int>();
stack.Push(42);            // Push onto stack
stack.Peek();              // View top element
stack.Pop();               // Pop from stack

// --- As a Queue (FIFO) ---
var queue = new Queue<int>();
queue.Enqueue(42);         // Enqueue (adds to back)
queue.Peek();              // View head
queue.Dequeue();           // Dequeue (removes from front)

deque.Count == 0;          // Check if empty
deque.Count;               // Current element count
```
