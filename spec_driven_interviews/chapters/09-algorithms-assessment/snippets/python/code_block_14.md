```python
import bisect

def medianSlidingWindow(nums: list[int], k: int) -> list[float]:
    window = sorted(nums[:k])
    result = []
    
    def get_median(w, k_val):
        if k_val % 2 == 1:
            return float(w[k_val // 2])
        return (w[k_val // 2 - 1] + w[k_val // 2]) / 2.0

    result.append(get_median(window, k))
    
    for i in range(k, len(nums)):
        # Remove old element
        window.remove(nums[i - k])
        # Insert new element
        bisect.insort(window, nums[i])
        result.append(get_median(window, k))
        
    return result
```