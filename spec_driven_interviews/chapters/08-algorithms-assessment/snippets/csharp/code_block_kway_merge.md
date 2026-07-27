```csharp
using System.Collections.Generic;

namespace AuraPay.Algorithms;

/// <summary>
/// Merges K sorted lists into one sorted list using a Min-Heap.
/// Time Complexity: O(N log K) where N is total elements, K is number of lists.
/// Space Complexity: O(K) auxiliary space for the heap.
/// </summary>
public class KWayMerge
{
    public class HeapNode
    {
        public int val;
        public int listIndex;
        public int elementIndex;

        public HeapNode(int val, int listIndex, int elementIndex)
        {
            this.val = val;
            this.listIndex = listIndex;
            this.elementIndex = elementIndex;
        }
    }

    /// <summary>
    /// Merges K sorted lists into a single sorted list.
    /// </summary>
    public List<int> MergeKLists(List<List<int>> lists)
    {
        // In C# .NET 6+, we can use PriorityQueue<TElement, TPriority>
        var minHeap = new PriorityQueue<HeapNode, int>();
        var result = new List<int>();

        // 1. Initialize heap with the first element of each list
        for (int i = 0; i < lists.Count; i++)
        {
            if (lists[i] != null && lists[i].Count > 0)
            {
                var node = new HeapNode(lists[i][0], i, 0);
                minHeap.Enqueue(node, node.val);
            }
        }

        // 2. Extract min and push the next element from that list
        while (minHeap.Count > 0)
        {
            var curr = minHeap.Dequeue();
            result.Add(curr.val);

            int nextElementIdx = curr.elementIndex + 1;
            if (nextElementIdx < lists[curr.listIndex].Count)
            {
                var node = new HeapNode(lists[curr.listIndex][nextElementIdx], curr.listIndex, nextElementIdx);
                minHeap.Enqueue(node, node.val);
            }
        }

        return result;
    }
}
```
