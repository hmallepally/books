```java
public int[] topKFrequent(int[] nums, int k) {
    var freqMap = new HashMap<Integer, Integer>();
    for (int n : nums) freqMap.merge(n, 1, Integer::sum);
    
    var minHeap = new PriorityQueue<Map.Entry<Integer, Integer>>(
        Comparator.comparingInt(Map.Entry::getValue));
    
    for (var entry : freqMap.entrySet()) {
        minHeap.offer(entry);
        if (minHeap.size() > k) minHeap.poll();
    }
    
    return minHeap.stream().mapToInt(Map.Entry::getKey).toArray();
}
```
