```java
public int[] sortByHeight(int[] a) {
    // Step 1: Extract all non-tree heights
    List<Integer> heights = new ArrayList<>();
    for (int h : a) {
        if (h != -1) heights.add(h);
    }

    // Step 2: Sort the extracted heights
    Collections.sort(heights);

    // Step 3: Reinsert sorted heights at non-tree positions
    int index = 0;
    for (int i = 0; i < a.length; i++) {
        if (a[i] != -1) {
            a[i] = heights.get(index++);
        }
    }

    return a;
}
// Time: O(N log N) for sorting, Space: O(N) for extracted list
```
