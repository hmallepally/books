```python
def subarraySum(nums: list[int], k: int) -> int:
    count = 0
    sum_val = 0
    prefix_sums = {0: 1}
    for num in nums:
        sum_val += num
        if (sum_val - k) in prefix_sums:
            count += prefix_sums[sum_val - k]
        prefix_sums[sum_val] = prefix_sums.get(sum_val, 0) + 1
    return count
```