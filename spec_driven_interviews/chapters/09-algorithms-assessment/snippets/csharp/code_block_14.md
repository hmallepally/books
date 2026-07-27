```csharp
public double[] MedianSlidingWindow(int[] nums, int k) {
    int n = nums.Length;
    double[] result = new double[n - k + 1];
    // Storing sorted array via simple list since C# SortedSet doesn't support duplicate values cleanly
    List<int> window = new List<int>();

    for (int i = 0; i < n; i++) {
        int val = nums[i];
        int insertPos = window.BinarySearch(val);
        if (insertPos < 0) insertPos = ~insertPos;
        window.Insert(insertPos, val);

        if (i >= k - 1) {
            if (k % 2 == 1) {
                result[i - k + 1] = window[k / 2];
            } else {
                result[i - k + 1] = ((double)window[k / 2 - 1] + window[k / 2]) / 2.0;
            }
            
            // Remove the element sliding out
            int elementToRemove = nums[i - k + 1];
            int removePos = window.BinarySearch(elementToRemove);
            window.RemoveAt(removePos);
        }
    }
    return result;
}
```