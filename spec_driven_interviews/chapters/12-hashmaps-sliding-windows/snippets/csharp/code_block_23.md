```csharp
public int[] TopKFrequent(int[] nums, int k) {
    Dictionary<int, int> count = new Dictionary<int, int>();
    foreach (int n in nums) count[n] = count.GetValueOrDefault(n, 0) + 1;
    PriorityQueue<int, int> heap = new PriorityQueue<int, int>();
    foreach (int n in count.Keys) {
        heap.Enqueue(n, count[n]);
        if (heap.Count > k) heap.Dequeue(); // Keep size K
    }
    int[] res = new int[k];
    for (int i = k - 1; i >= 0; i--) res[i] = heap.Dequeue();
    return res;
}
// Time Complexity: O(N log K) | Space Complexity: O(N)
```