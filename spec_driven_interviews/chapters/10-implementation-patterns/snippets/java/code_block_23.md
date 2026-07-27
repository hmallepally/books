```java
public int adjacentElementsProduct(int[] inputArray) {
    if (inputArray == null || inputArray.length < 2) return 0;

    int maxProd = inputArray[0] * inputArray[1];

    for (int i = 1; i < inputArray.length - 1; i++) {
        int prod = inputArray[i] * inputArray[i + 1];
        if (prod > maxProd) {
            maxProd = prod;
        }
    }

    return maxProd;
}
// Time: O(N), Space: O(1)
```
