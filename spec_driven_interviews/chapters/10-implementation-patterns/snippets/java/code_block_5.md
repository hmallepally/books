```java
public boolean isPalindrome(String s) {
    if (s == null) return false;

    int left = 0, right = s.length() - 1;

    while (left < right) {
        // Skip non-alphanumeric from the left
        while (left < right && !Character.isLetterOrDigit(s.charAt(left))) {
            left++;
        }
        // Skip non-alphanumeric from the right
        while (left < right && !Character.isLetterOrDigit(s.charAt(right))) {
            right--;
        }

        // Compare characters (case-insensitive)
        if (Character.toLowerCase(s.charAt(left)) != Character.toLowerCase(s.charAt(right))) {
            return false;
        }

        left++;
        right--;
    }

    return true;
}
// Time: O(N), Space: O(1)
```
