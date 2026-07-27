```csharp
public int TotalFruit(int[] fruits) {
    Dictionary<int, int> count = new Dictionary<int, int>();
    int left = 0, max = 0;
    for (int right = 0; right < fruits.Length; right++) {
        count[fruits[right]] = count.GetValueOrDefault(fruits[right], 0) + 1;
        while (count.Count > 2) {
            count[fruits[left]]--;
            if (count[fruits[left]] == 0) count.Remove(fruits[left]);
            left++;
        }
        max = Math.Max(max, right - left + 1);
    }
    return max;
}
// Time Complexity: O(N) | Space Complexity: O(1)
```