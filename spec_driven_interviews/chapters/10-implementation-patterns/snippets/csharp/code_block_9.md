```csharp
public bool IsValid(string s) {
    if (s == null || s.Length % 2 != 0) return false;

    char[] stack = new char[s.Length];
    int top = -1;

    foreach (char c in s) {
        if (c == '(') stack[++top] = ')';
        else if (c == '{') stack[++top] = '}';
        else if (c == '[') stack[++top] = ']';
        else {
            if (top == -1 || stack[top--] != c) return false;
        }
    }

    return top == -1; // Stack must be empty
}
// Time: O(N), Space: O(N) worst case for the stack
```