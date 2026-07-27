```java
public int minMeetingRooms(int[][] intervals) {
    if (intervals == null || intervals.length == 0) return 0;
    Arrays.sort(intervals, (a, b) -> Integer.compare(a[0], b[0]));

    var minHeap = new PriorityQueue<Integer>(); // Stores end times
    minHeap.offer(intervals[0][1]);

    for (int i = 1; i < intervals.length; i++) {
        if (intervals[i][0] >= minHeap.peek()) {
            minHeap.poll(); // Room freed up!
        }
        minHeap.offer(intervals[i][1]); // Allocate room
    }
    return minHeap.size();
}
```
