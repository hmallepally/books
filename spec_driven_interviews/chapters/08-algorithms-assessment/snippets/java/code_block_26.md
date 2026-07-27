```java
Deque<Integer> deque = new ArrayDeque<>();

// --- As a double-ended queue ---
deque.offerFirst(1);       // Add to front — O(1)
deque.offerLast(2);        // Add to back — O(1)
deque.peekFirst();         // View front without removing — O(1)
deque.peekLast();          // View back without removing — O(1)
deque.pollFirst();         // Remove from front — O(1)
deque.pollLast();          // Remove from back — O(1)

// --- As a Stack (LIFO) — use instead of java.util.Stack ---
deque.push(42);            // Push onto stack (adds to front)
deque.peek();              // View top element
deque.pop();               // Pop from stack (removes from front)

// --- As a Queue (FIFO) ---
deque.offer(42);           // Enqueue (adds to back)
deque.peek();              // View head
deque.poll();              // Dequeue (removes from front)

deque.isEmpty();           // Check if empty
deque.size();              // Current element count
```
