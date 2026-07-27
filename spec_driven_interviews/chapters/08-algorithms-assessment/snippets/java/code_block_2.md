```java
package com.aurapay.algorithms;

import java.util.Arrays;
import java.util.Comparator;

/**
 * Solves the Interval Scheduling problem using a Greedy approach.
 * Minimizes overlaps, finding the maximum number of non-overlapping transactions 
 * that can be processed.
 * Time Complexity: O(N log N) due to sorting.
 * Space Complexity: O(1) or O(N) depending on sort implementation.
 */
public class IntervalScheduler {

    public static class TransactionInterval {
        int start;
        int end;

        public TransactionInterval(int start, int end) {
            this.start = start;
            this.end = end;
        }
    }

    /**
     * Returns the maximum number of non-overlapping transaction intervals.
     */
    public int maxNonOverlappingTransactions(TransactionInterval[] intervals) {
        if (intervals == null || intervals.length == 0) {
            return 0;
        }

        // GREEDY INVARIANT: Sort intervals by their end time.
        // Selecting the interval that ends earliest leaves the maximum space 
        // for subsequent transactions.
        Arrays.sort(intervals, Comparator.comparingInt(a -> a.end));

        int count = 1; // Always select the first interval
        int lastSelectedEnd = intervals[0].end;

        for (int i = 1; i < intervals.length; i++) {
            // If the start time is greater than or equal to the end time of the 
            // last selected interval, select this transaction
            if (intervals[i].start >= lastSelectedEnd) {
                count++;
                lastSelectedEnd = intervals[i].end;
            }
        }

        return count;
    }
}
```
