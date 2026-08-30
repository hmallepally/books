"""
Module: 3-Step Sliding Window & Invariant Patterns
Corresponds to Chapters 01, 10, and 12.
"""
from typing import List, Dict

def length_of_longest_substring_k_distinct(s: str, k: int) -> int:
    """
    Finds the length of the longest substring with at most k distinct characters.
    
    Invariant: At all points, the window s[left:right+1] contains <= k distinct characters.
    Time Complexity: O(N)
    Space Complexity: O(K)
    """
    if k == 0 or not s:
        return 0
    
    freq_map: Dict[str, int] = {}
    max_length = 0
    left = 0
    
    for right, char in enumerate(s):
        # 1. Expand right boundary and ingest state
        freq_map[char] = freq_map.get(char, 0) + 1
        
        # 2. Contract left boundary while invariant is violated
        while len(freq_map) > k:
            left_char = s[left]
            freq_map[left_char] -= 1
            if freq_map[left_char] == 0:
                del freq_map[left_char]
            left += 1
            
        # 3. Record optimal valid window metric
        max_length = max(max_length, right - left + 1)
        
    return max_length


def subarray_sum_equals_k(nums: List[int], k: int) -> int:
    """
    Counts total contiguous subarrays whose sum equals exactly k.
    
    Uses Combinatorial Prefix Sum Contribution Counting with negative number support.
    Time Complexity: O(N)
    Space Complexity: O(N)
    """
    prefix_count: Dict[int, int] = {0: 1} # Base case: empty prefix
    current_sum = 0
    total_subarrays = 0
    
    for x in nums:
        current_sum += x
        target = current_sum - k
        if target in prefix_count:
            total_subarrays += prefix_count[target]
            
        prefix_count[current_sum] = prefix_count.get(current_sum, 0) + 1
        
    return total_subarrays
