```java
public void reverseString(char[] s) {
    if (s == null || s.length <= 1) return;

    int left = 0, right = s.length - 1;
    while (left < right) {
        char temp = s[left];
        s[left] = s[right];
        s[right] = temp;
        left++;
        right--;
    }
}
// Time: O(N), Space: O(1)
```
