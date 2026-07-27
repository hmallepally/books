```java
public int leastInterval(char[] tasks, int n) {
    int[] count = new int[26];
    int max = 0, maxCount = 0;
    for (char c : tasks) {
        count[c - 'A']++;
        if (count[c - 'A'] == max) maxCount++;
        else if (count[c - 'A'] > max) { max = count[c - 'A']; maxCount = 1; }
    }
    int emptySlots = (max - 1) * (n - (maxCount - 1));
    int availableTasks = tasks.length - max * maxCount;
    int idles = Math.max(0, emptySlots - availableTasks);
    return tasks.length + idles;
}
// Time Complexity: O(N) | Space Complexity: O(1)
```
