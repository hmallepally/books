```python
def compress(self, chars: list[str]) -> int:
    if not chars:
        return 0

    write = 0 # Write pointer for compressed output
    read = 0  # Read pointer scanning input

    while read < len(chars):
        current = chars[read]
        count = 0

        # Count consecutive occurrences of current character
        while read < len(chars) and chars[read] == current:
            read += 1
            count += 1

        # Write the character itself
        chars[write] = current
        write += 1

        # Write the count digits (only if count > 1)
        if count > 1:
            # Convert count to individual digit characters
            for digit in str(count):
                chars[write] = digit
                write += 1

    return write
# Time: O(N), Space: O(1) auxiliary
```