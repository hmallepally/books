```python
from typing import List

class TransactionInterval:
    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end

class IntervalScheduler:
    """
    Solves the Interval Scheduling problem using a Greedy approach.
    """
    def max_non_overlapping_transactions(self, intervals: List[TransactionInterval]) -> int:
        if not intervals:
            return 0

        # GREEDY INVARIANT: Sort intervals by their end time.
        sorted_intervals = sorted(intervals, key=lambda x: x.end)

        count = 1
        last_selected_end = sorted_intervals[0].end

        for i in range(1, len(sorted_intervals)):
            # If start time is greater than or equal to last selected end time
            if sorted_intervals[i].start >= last_selected_end:
                count += 1
                last_selected_end = sorted_intervals[i].end

        return count
```
