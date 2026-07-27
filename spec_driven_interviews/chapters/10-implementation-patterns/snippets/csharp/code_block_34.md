```csharp
public string ReverseInParentheses(string s) {
    Stack<StringBuilder> stack = new Stack<StringBuilder>();
    stack.Push(new StringBuilder());

    foreach (char c in s) {
        if (c == '(') {
            stack.Push(new StringBuilder()); // Start new nested context
        } else if (c == ')') {
            StringBuilder inner = stack.Pop();  // Pop innermost context
            
            // Reverse the inner StringBuilder
            char[] innerChars = inner.ToString().ToCharArray();
            Array.Reverse(innerChars);
            
            stack.Peek().Append(innerChars); // Append to enclosing context
        } else {
            stack.Peek().Append(c);          // Accumulate character
        }
    }

    return stack.Peek().ToString();
}
// Time: O(N^2) worst case for nested reversals, Space: O(N)
```