```csharp
public int[][] Merge(int[][] intervals) {
    if (intervals.Length <= 1) return intervals;
    Array.Sort(intervals, (a, b) => a[0].CompareTo(b[0]));
    var merged = new List<int[]>();
    int[] current = intervals[0];
    merged.Add(current);
    foreach (var interval in intervals) {
        if (interval[0] <= current[1]) {
            current[1] = Math.Max(current[1], interval[1]);
        } else {
            current = interval;
            merged.Add(current);
        }
    }
    return merged.ToArray();
}
```