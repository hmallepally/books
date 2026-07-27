```csharp
public int[] SortByHeight(int[] a) {
    // Step 1: Extract all non-tree heights
    List<int> heights = new List<int>();
    foreach (int h in a) {
        if (h != -1) heights.Add(h);
    }

    // Step 2: Sort the extracted heights
    heights.Sort();

    // Step 3: Reinsert sorted heights at non-tree positions
    int index = 0;
    for (int i = 0; i < a.Length; i++) {
        if (a[i] != -1) {
            a[i] = heights[index++];
        }
    }

    return a;
}
// Time: O(N log N) for sorting, Space: O(N) for extracted list
```