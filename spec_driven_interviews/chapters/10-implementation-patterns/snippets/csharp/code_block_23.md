```csharp
public int AdjacentElementsProduct(int[] inputArray) {
    if (inputArray == null || inputArray.Length < 2) return 0;

    int maxProd = inputArray[0] * inputArray[1];

    for (int i = 1; i < inputArray.Length - 1; i++) {
        int prod = inputArray[i] * inputArray[i + 1];
        if (prod > maxProd) {
            maxProd = prod;
        }
    }

    return maxProd;
}
// Time: O(N), Space: O(1)
```