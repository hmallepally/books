```csharp
public int LeastInterval(char[] tasks, int n) {
    int[] count = new int[26];
    int max = 0, maxCount = 0;
    foreach (char c in tasks) {
        count[c - 'A']++;
        if (count[c - 'A'] == max) maxCount++;
        else if (count[c - 'A'] > max) { max = count[c - 'A']; maxCount = 1; }
    }
    int emptySlots = (max - 1) * (n - (maxCount - 1));
    int availableTasks = tasks.Length - max * maxCount;
    int idles = Math.Max(0, emptySlots - availableTasks);
    return tasks.Length + idles;
}
// Time Complexity: O(N) | Space Complexity: O(1)
```