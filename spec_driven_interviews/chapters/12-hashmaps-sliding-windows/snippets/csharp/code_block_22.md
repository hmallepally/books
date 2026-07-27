```csharp
public int[][] Insert(int[][] intervals, int[] newInterval) {
    List<int[]> res = new List<int[]>();
    int i = 0, n = intervals.Length;
    while (i < n && intervals[i][1] < newInterval[0]) res.Add(intervals[i++]); // Before
    while (i < n && intervals[i][0] <= newInterval[1]) { // Merge
        newInterval[0] = Math.Min(newInterval[0], intervals[i][0]);
        newInterval[1] = Math.Max(newInterval[1], intervals[i][1]);
        i++;
    }
    res.Add(newInterval);
    while (i < n) res.Add(intervals[i++]); // After
    return res.ToArray();
}
// Time Complexity: O(N) | Space Complexity: O(N)
```