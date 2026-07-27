```csharp
using System;
using System.Collections.Generic;

namespace AuraPay.Algorithms
{
    public class TransactionInterval
    {
        public int Start { get; }
        public int End { get; }

        public TransactionInterval(int start, int end)
        {
            Start = start;
            End = end;
        }
    }

    /// <summary>
    /// Solves the Interval Scheduling problem using a Greedy approach.
    /// </summary>
    public class IntervalScheduler
    {
        public int MaxNonOverlappingTransactions(TransactionInterval[] intervals)
        {
            if (intervals == null || intervals.Length == 0)
            {
                return 0;
            }

            // GREEDY INVARIANT: Sort intervals by their end time.
            Array.Sort(intervals, (a, b) => a.End.CompareTo(b.End));

            int count = 1;
            int lastSelectedEnd = intervals[0].End;

            for (int i = 1; i < intervals.Length; i++)
            {
                // If the start time is greater than or equal to the end time of the 
                // last selected interval, select this transaction
                if (intervals[i].Start >= lastSelectedEnd)
                {
                    count++;
                    lastSelectedEnd = intervals[i].End;
                }
            }

            return count;
        }
    }
}
```
