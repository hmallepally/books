```csharp
public int FindKthLargest(int[] nums, int k) {
    var minHeap = new PriorityQueue<int, int>();
    foreach (int num in nums) {
        minHeap.Enqueue(num, num);
        if (minHeap.Count > k) {
            minHeap.Dequeue();
        }
    }
    return minHeap.Peek();
}
```
