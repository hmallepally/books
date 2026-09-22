```java
int k = 3, sum = 0, max = 0; // <1>
for (int i = 0; i < arr.length; i++) { // <2>
    sum += arr[i]; // <3>
    if (i >= k - 1) { // <4>
        max = Math.max(max, sum);
        sum -= arr[i - (k - 1)]; // <5>
    }
}
```
