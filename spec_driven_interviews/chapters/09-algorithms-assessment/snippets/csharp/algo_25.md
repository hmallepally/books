```csharp
public int[] TopKFrequent(int[] nums, int k)
{
    var freqMap = new Dictionary<int, int>();
    foreach (int n in nums)
    {
        freqMap[n] = freqMap.GetValueOrDefault(n, 0) + 1;
    }

    var minHeap = new PriorityQueue<int, int>();

    foreach (var entry in freqMap)
    {
        minHeap.Enqueue(entry.Key, entry.Value);
        if (minHeap.Count > k) minHeap.Dequeue();
    }

    var result = new int[k];
    for (int i = 0; i < k; i++)
    {
        result[i] = minHeap.Dequeue();
    }
    return result;
}
```
