```java
package com.aurapay.algorithms;

import java.util.ArrayList;
import java.util.List;
import java.util.PriorityQueue;

/**
 * Merges K sorted lists into one sorted list using a Min-Heap.
 * Time Complexity: O(N log K) where N is total elements, K is number of lists.
 * Space Complexity: O(K) auxiliary space for the heap.
 */
public class KWayMerge {

    public static class HeapNode {
        int val;
        int listIndex;
        int elementIndex;

        HeapNode(int val, int listIndex, int elementIndex) {
            this.val = val;
            this.listIndex = listIndex;
            this.elementIndex = elementIndex;
        }
    }

    /**
     * Merges K sorted lists into a single sorted list.
     */
    public List<Integer> mergeKLists(List<List<Integer>> lists) {
        PriorityQueue<HeapNode> minHeap = new PriorityQueue<>((a, b) -> a.val - b.val);
        List<Integer> result = new ArrayList<>();

        // 1. Initialize heap with the first element of each list
        for (int i = 0; i < lists.size(); i++) {
            if (lists.get(i) != null && !lists.get(i).isEmpty()) {
                minHeap.offer(new HeapNode(lists.get(i).get(0), i, 0));
            }
        }

        // 2. Repeatedly extract the minimum value and push the next element from that list
        while (!minHeap.isEmpty()) {
            HeapNode curr = minHeap.poll();
            result.add(curr.val);

            int nextElementIdx = curr.elementIndex + 1;
            if (nextElementIdx < lists.get(curr.listIndex).size()) {
                minHeap.offer(new HeapNode(
                    lists.get(curr.listIndex).get(nextElementIdx),
                    curr.listIndex,
                    nextElementIdx
                ));
            }
        }

        return result;
    }
}
```
