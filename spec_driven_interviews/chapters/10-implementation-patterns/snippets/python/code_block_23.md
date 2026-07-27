```python
def adjacent_elements_product(self, input_array: list[int]) -> int:
    if not input_array or len(input_array) < 2:
        return 0

    max_prod = input_array[0] * input_array[1]

    for i in range(1, len(input_array) - 1):
        prod = input_array[i] * input_array[i + 1]
        if prod > max_prod:
            max_prod = prod

    return max_prod
# Time: O(N), Space: O(1)
```