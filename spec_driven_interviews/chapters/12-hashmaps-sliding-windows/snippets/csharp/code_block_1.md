```csharp
int left = 0, maxLen = 0;
for (int right = 0; right < arr.Length; right++) {
    // 1. Add arr[right] to window state
    while (false /* window state violates invariant */) {
        // 2. Remove arr[left] from window state
        left++;
    }
    // 3. Update maxLen or minLen
    maxLen = Math.Max(maxLen, right - left + 1);
}
```