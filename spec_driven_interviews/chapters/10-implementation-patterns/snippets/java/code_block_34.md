```java
public String reverseInParentheses(String s) {
    Deque<StringBuilder> stack = new ArrayDeque<>();
    stack.push(new StringBuilder());

    for (char c : s.toCharArray()) {
        if (c == '(') {
            stack.push(new StringBuilder()); // Start new nested context
        } else if (c == ')') {
            StringBuilder inner = stack.pop();  // Pop innermost context
            inner.reverse();                     // Reverse it
            stack.peek().append(inner);          // Append to enclosing context
        } else {
            stack.peek().append(c);              // Accumulate character
        }
    }

    return stack.peek().toString();
}
// Time: O(N^2) worst case for nested reversals, Space: O(N)
```
