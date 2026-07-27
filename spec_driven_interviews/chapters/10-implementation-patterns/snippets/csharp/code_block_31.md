```csharp
public int ArrayChange(int[] inputArray) {
    int moves = 0;

    for (int i = 1; i < inputArray.Length; i++) {
        if (inputArray[i] <= inputArray[i - 1]) {
            // Calculate the minimum increment needed
            int deficit = inputArray[i - 1] - inputArray[i] + 1;
            inputArray[i] += deficit;
            moves += deficit;
        }
    }

    return moves;
}
// Time: O(N), Space: O(1)
```