```python
def array_change(self, input_array: list[int]) -> int:
    moves = 0

    for i in range(1, len(input_array)):
        if input_array[i] <= input_array[i - 1]:
            # Calculate the minimum increment needed
            deficit = input_array[i - 1] - input_array[i] + 1
            input_array[i] += deficit
            moves += deficit

    return moves
# Time: O(N), Space: O(1)
```