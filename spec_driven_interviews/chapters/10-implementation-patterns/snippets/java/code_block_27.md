```java
public boolean isLucky(int n) {
    String s = String.valueOf(n);
    int mid = s.length() / 2;
    int sum1 = 0, sum2 = 0;

    for (int i = 0; i < mid; i++) {
        sum1 += s.charAt(i) - '0';       // First half digit
        sum2 += s.charAt(i + mid) - '0'; // Second half digit
    }

    return sum1 == sum2;
}
// Time: O(D) where D is digit count, Space: O(D) for string conversion
```
