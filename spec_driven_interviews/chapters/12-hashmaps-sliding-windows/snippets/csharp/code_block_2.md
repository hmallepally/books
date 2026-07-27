```csharp
int k = 3, sum = 0, max = 0;
for (int i = 0; i < arr.Length; i++) {
    sum += arr[i]; // Add current element
    if (i >= k - 1) {
        max = Math.Max(max, sum); // Update result
        sum -= arr[i - (k - 1)];  // Remove leftmost element for next iteration
    }
}
```