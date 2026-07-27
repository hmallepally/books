import os

translations = {
    "1": {
        "python": """```python
left = max_len = 0
for right in range(len(arr)):
    # 1. Add arr[right] to window state
    while False: # window state violates invariant
        # 2. Remove arr[left] from window state
        left += 1
    # 3. Update maxLen or minLen
    max_len = max(max_len, right - left + 1)
```""",
        "csharp": """```csharp
int left = 0, maxLen = 0;
for (int right = 0; right < arr.Length; right++) {
    // 1. Add arr[right] to window state
    while (false /* window state violates invariant */) {
        // 2. Remove arr[left] from window state
        left++;
    }
    // 3. Update maxLen or minLen
    maxLen = Math.Max(maxLen, right - left + 1);
}
```"""
    },
    "2": {
        "python": """```python
k, total_sum, max_val = 3, 0, 0
for i in range(len(arr)):
    total_sum += arr[i] # Add current element
    if i >= k - 1:
        max_val = max(max_val, total_sum) # Update result
        total_sum -= arr[i - (k - 1)]     # Remove leftmost element for next iteration
```""",
        "csharp": """```csharp
int k = 3, sum = 0, max = 0;
for (int i = 0; i < arr.Length; i++) {
    sum += arr[i]; // Add current element
    if (i >= k - 1) {
        max = Math.Max(max, sum); // Update result
        sum -= arr[i - (k - 1)];  // Remove leftmost element for next iteration
    }
}
```"""
    },
    "3": {
        "python": """```python
from collections import defaultdict
hash_map = defaultdict(int)
hash_map[0] = 1 # Base case for subarrays starting at index 0
total_sum = count = 0
for num in nums:
    total_sum += num
    if (total_sum - k) in hash_map:
        count += hash_map[total_sum - k]
    hash_map[total_sum] += 1
```""",
        "csharp": """```csharp
Dictionary<int, int> map = new Dictionary<int, int>();
map[0] = 1; // Base case for subarrays starting at index 0
int sum = 0, count = 0;
foreach (int num in nums) {
    sum += num;
    if (map.ContainsKey(sum - k)) {
        count += map[sum - k];
    }
    map[sum] = map.GetValueOrDefault(sum, 0) + 1;
}
```"""
    },
    "4": {
        "python": """```python
from collections import defaultdict
hash_map = defaultdict(list)
for s in strs:
    count = [0] * 26
    for c in s: count[ord(c) - ord('a')] += 1
    key = str(count)
    hash_map[key].append(s)
```""",
        "csharp": """```csharp
Dictionary<string, List<string>> map = new Dictionary<string, List<string>>();
foreach (string s in strs) {
    int[] count = new int[26];
    foreach (char c in s.ToCharArray()) count[c - 'a']++;
    string key = string.Join(",", count);
    if (!map.ContainsKey(key)) map[key] = new List<string>();
    map[key].Add(s);
}
```"""
    },
    "5": {
        "python": """```python
def length_of_longest_substring(self, s: str) -> int:
    char_set = set()
    left = max_val = 0
    for right in range(len(s)):
        # Contract if duplicate found
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1
        char_set.add(s[right]) # Add current char
        max_val = max(max_val, right - left + 1)
    return max_val
# Time Complexity: O(N) | Space Complexity: O(min(N, M))
```""",
        "csharp": """```csharp
public int LengthOfLongestSubstring(string s) {
    HashSet<char> set = new HashSet<char>();
    int left = 0, max = 0;
    for (int right = 0; right < s.Length; right++) {
        // Contract if duplicate found
        while (set.Contains(s[right])) {
            set.Remove(s[left++]);
        }
        set.Add(s[right]); // Add current char
        max = Math.Max(max, right - left + 1);
    }
    return max;
}
// Time Complexity: O(N) | Space Complexity: O(min(N, M))
```"""
    },
    "6": {
        "python": """```python
def subarray_sum(self, nums: list[int], k: int) -> int:
    from collections import defaultdict
    hash_map = defaultdict(int)
    hash_map[0] = 1 # Base case
    total_sum = count = 0
    for num in nums:
        total_sum += num
        # Check if required prefix exists
        if (total_sum - k) in hash_map: count += hash_map[total_sum - k]
        hash_map[total_sum] += 1
    return count
# Time Complexity: O(N) | Space Complexity: O(N)
```""",
        "csharp": """```csharp
public int SubarraySum(int[] nums, int k) {
    Dictionary<int, int> map = new Dictionary<int, int>();
    map[0] = 1; // Base case
    int sum = 0, count = 0;
    foreach (int num in nums) {
        sum += num;
        // Check if required prefix exists
        if (map.ContainsKey(sum - k)) count += map[sum - k];
        map[sum] = map.GetValueOrDefault(sum, 0) + 1;
    }
    return count;
}
// Time Complexity: O(N) | Space Complexity: O(N)
```"""
    },
    "7": {
        "python": """```python
def group_anagrams(self, strs: list[str]) -> list[list[str]]:
    from collections import defaultdict
    hash_map = defaultdict(list)
    for s in strs:
        count = [0] * 26
        for c in s: count[ord(c) - ord('a')] += 1 # Build signature
        key = tuple(count)
        hash_map[key].append(s)
    return list(hash_map.values())
# Time Complexity: O(N * L) | Space Complexity: O(N * L)
```""",
        "csharp": """```csharp
public IList<IList<string>> GroupAnagrams(string[] strs) {
    Dictionary<string, List<string>> map = new Dictionary<string, List<string>>();
    foreach (string s in strs) {
        int[] count = new int[26];
        foreach (char c in s) count[c - 'a']++; // Build signature
        string key = string.Join(",", count);
        if (!map.ContainsKey(key)) map[key] = new List<string>();
        map[key].Add(s);
    }
    return new List<IList<string>>(map.Values);
}
// Time Complexity: O(N * L) | Space Complexity: O(N * L)
```"""
    },
    "8": {
        "python": """```python
def find_anagrams(self, s: str, p: str) -> list[int]:
    res = []
    if len(s) < len(p): return res
    p_count, s_count = [0] * 26, [0] * 26
    for c in p: p_count[ord(c) - ord('a')] += 1
    for i in range(len(s)):
        s_count[ord(s[i]) - ord('a')] += 1
        if i >= len(p): s_count[ord(s[i - len(p)]) - ord('a')] -= 1 # Contract
        if p_count == s_count: res.append(i - len(p) + 1) # Match
    return res
# Time Complexity: O(N) | Space Complexity: O(1)
```""",
        "csharp": """```csharp
public IList<int> FindAnagrams(string s, string p) {
    List<int> res = new List<int>();
    if (s.Length < p.Length) return res;
    int[] pCount = new int[26], sCount = new int[26];
    foreach (char c in p) pCount[c - 'a']++;
    for (int i = 0; i < s.Length; i++) {
        sCount[s[i] - 'a']++;
        if (i >= p.Length) sCount[s[i - p.Length] - 'a']--; // Contract
        if (pCount.SequenceEqual(sCount)) res.Add(i - p.Length + 1); // Match
    }
    return res;
}
// Time Complexity: O(N) | Space Complexity: O(1)
```"""
    },
    "9": {
        "python": """```python
def length_of_longest_substring_k_distinct(self, s: str, k: int) -> int:
    from collections import defaultdict
    hash_map = defaultdict(int)
    left = max_val = 0
    for right in range(len(s)):
        c = s[right]
        hash_map[c] += 1
        while len(hash_map) > k: # Invariant broken
            left_char = s[left]
            left += 1
            hash_map[left_char] -= 1
            if hash_map[left_char] == 0: del hash_map[left_char]
        max_val = max(max_val, right - left + 1)
    return max_val
# Time Complexity: O(N) | Space Complexity: O(K)
```""",
        "csharp": """```csharp
public int LengthOfLongestSubstringKDistinct(string s, int k) {
    Dictionary<char, int> map = new Dictionary<char, int>();
    int left = 0, max = 0;
    for (int right = 0; right < s.Length; right++) {
        char c = s[right];
        map[c] = map.GetValueOrDefault(c, 0) + 1;
        while (map.Count > k) { // Invariant broken
            char leftChar = s[left++];
            map[leftChar]--;
            if (map[leftChar] == 0) map.Remove(leftChar);
        }
        max = Math.Max(max, right - left + 1);
    }
    return max;
}
// Time Complexity: O(N) | Space Complexity: O(K)
```"""
    },
    "10": {
        "python": """```python
def min_window(self, s: str, t: str) -> str:
    char_map = [0] * 128
    for c in t: char_map[ord(c)] += 1
    left, count = 0, len(t)
    min_len, min_start = float('inf'), 0
    
    for right in range(len(s)):
        if char_map[ord(s[right])] > 0: count -= 1 # Found required char
        char_map[ord(s[right])] -= 1
        
        while count == 0: # All chars found
            if right - left + 1 < min_len:
                min_len = right - left + 1
                min_start = left
            char_map[ord(s[left])] += 1
            if char_map[ord(s[left])] > 0: count += 1 # Removed required char
            left += 1
            
    return "" if min_len == float('inf') else s[min_start:min_start + min_len]
# Time Complexity: O(N) | Space Complexity: O(1)
```""",
        "csharp": """```csharp
public string MinWindow(string s, string t) {
    int[] map = new int[128];
    foreach (char c in t) map[c]++;
    int left = 0, count = t.Length, minLen = int.MaxValue, minStart = 0;
    for (int right = 0; right < s.Length; right++) {
        if (map[s[right]]-- > 0) count--; // Found required char
        while (count == 0) { // All chars found
            if (right - left + 1 < minLen) {
                minLen = right - left + 1;
                minStart = left;
            }
            if (++map[s[left++]] > 0) count++; // Removed required char
        }
    }
    return minLen == int.MaxValue ? "" : s.Substring(minStart, minLen);
}
// Time Complexity: O(N) | Space Complexity: O(1)
```"""
    },
    "11": {
        "python": """```python
def group_strings(self, strings: list[str]) -> list[list[str]]:
    from collections import defaultdict
    hash_map = defaultdict(list)
    for s in strings:
        key = []
        for i in range(1, len(s)):
            diff = (ord(s[i]) - ord(s[i-1]) + 26) % 26 # Circular difference
            key.append(str(diff))
        hash_map[','.join(key)].append(s)
    return list(hash_map.values())
# Time Complexity: O(N * L) | Space Complexity: O(N * L)
```""",
        "csharp": """```csharp
public IList<IList<string>> GroupStrings(string[] strings) {
    Dictionary<string, List<string>> map = new Dictionary<string, List<string>>();
    foreach (string s in strings) {
        StringBuilder key = new StringBuilder();
        for (int i = 1; i < s.Length; i++) {
            int diff = (s[i] - s[i-1] + 26) % 26; // Circular difference
            key.Append(diff).Append(",");
        }
        string k = key.ToString();
        if (!map.ContainsKey(k)) map[k] = new List<string>();
        map[k].Add(s);
    }
    return new List<IList<string>>(map.Values);
}
// Time Complexity: O(N * L) | Space Complexity: O(N * L)
```"""
    },
    "12": {
        "python": """```python
def find_max_length(self, nums: list[int]) -> int:
    hash_map = {0: -1}
    total_sum = max_val = 0
    for i, num in enumerate(nums):
        total_sum += -1 if num == 0 else 1 # Map 0 to -1
        if total_sum in hash_map:
            max_val = max(max_val, i - hash_map[total_sum])
        else:
            hash_map[total_sum] = i # Store first occurrence
    return max_val
# Time Complexity: O(N) | Space Complexity: O(N)
```""",
        "csharp": """```csharp
public int FindMaxLength(int[] nums) {
    Dictionary<int, int> map = new Dictionary<int, int>();
    map[0] = -1;
    int sum = 0, max = 0;
    for (int i = 0; i < nums.Length; i++) {
        sum += nums[i] == 0 ? -1 : 1; // Map 0 to -1
        if (map.ContainsKey(sum)) {
            max = Math.Max(max, i - map[sum]);
        } else {
            map[sum] = i; // Store first occurrence
        }
    }
    return max;
}
// Time Complexity: O(N) | Space Complexity: O(N)
```"""
    },
    "13": {
        "python": """```python
def num_subarray_product_less_than_k(self, nums: list[int], k: int) -> int:
    if k <= 1: return 0
    prod, left, count = 1, 0, 0
    for right in range(len(nums)):
        prod *= nums[right]
        while prod >= k:
            prod //= nums[left]
            left += 1 # Shrink
        count += right - left + 1 # Add valid subarrays
    return count
# Time Complexity: O(N) | Space Complexity: O(1)
```""",
        "csharp": """```csharp
public int NumSubarrayProductLessThanK(int[] nums, int k) {
    if (k <= 1) return 0;
    int prod = 1, left = 0, count = 0;
    for (int right = 0; right < nums.Length; right++) {
        prod *= nums[right];
        while (prod >= k) prod /= nums[left++]; // Shrink
        count += right - left + 1; // Add valid subarrays
    }
    return count;
}
// Time Complexity: O(N) | Space Complexity: O(1)
```"""
    },
    "14": {
        "python": """```python
def check_inclusion(self, s1: str, s2: str) -> bool:
    if len(s1) > len(s2): return False
    s1_map, s2_map = [0] * 26, [0] * 26
    for c in s1: s1_map[ord(c) - ord('a')] += 1
    for i in range(len(s2)):
        s2_map[ord(s2[i]) - ord('a')] += 1
        if i >= len(s1): s2_map[ord(s2[i - len(s1)]) - ord('a')] -= 1
        if s1_map == s2_map: return True
    return False
# Time Complexity: O(N) | Space Complexity: O(1)
```""",
        "csharp": """```csharp
public bool CheckInclusion(string s1, string s2) {
    if (s1.Length > s2.Length) return false;
    int[] s1map = new int[26], s2map = new int[26];
    foreach (char c in s1) s1map[c - 'a']++;
    for (int i = 0; i < s2.Length; i++) {
        s2map[s2[i] - 'a']++;
        if (i >= s1.Length) s2map[s2[i - s1.Length] - 'a']--;
        if (s1map.SequenceEqual(s2map)) return true;
    }
    return false;
}
// Time Complexity: O(N) | Space Complexity: O(1)
```"""
    },
    "15": {
        "python": """```python
def maximum_unique_subarray(self, nums: list[int]) -> int:
    char_set = set()
    total_sum = max_val = left = 0
    for right in range(len(nums)):
        while nums[right] in char_set:
            char_set.remove(nums[left])
            total_sum -= nums[left] # Remove duplicate
            left += 1
        char_set.add(nums[right])
        total_sum += nums[right]
        max_val = max(max_val, total_sum)
    return max_val
# Time Complexity: O(N) | Space Complexity: O(N)
```""",
        "csharp": """```csharp
public int MaximumUniqueSubarray(int[] nums) {
    HashSet<int> set = new HashSet<int>();
    int sum = 0, max = 0, left = 0;
    for (int right = 0; right < nums.Length; right++) {
        while (set.Contains(nums[right])) {
            set.Remove(nums[left]);
            sum -= nums[left++]; // Remove duplicate
        }
        set.Add(nums[right]);
        sum += nums[right];
        max = Math.Max(max, sum);
    }
    return max;
}
// Time Complexity: O(N) | Space Complexity: O(N)
```"""
    },
    "16": {
        "python": """```python
def character_replacement(self, s: str, k: int) -> int:
    count = [0] * 26
    max_count = left = max_len = 0
    for right in range(len(s)):
        idx = ord(s[right]) - ord('A')
        count[idx] += 1
        max_count = max(max_count, count[idx])
        if right - left + 1 - max_count > k: # Invalid window
            count[ord(s[left]) - ord('A')] -= 1
            left += 1
        max_len = max(max_len, right - left + 1)
    return max_len
# Time Complexity: O(N) | Space Complexity: O(1)
```""",
        "csharp": """```csharp
public int CharacterReplacement(string s, int k) {
    int[] count = new int[26];
    int maxCount = 0, left = 0, maxLen = 0;
    for (int right = 0; right < s.Length; right++) {
        maxCount = Math.Max(maxCount, ++count[s[right] - 'A']);
        if (right - left + 1 - maxCount > k) { // Invalid window
            count[s[left++] - 'A']--;
        }
        maxLen = Math.Max(maxLen, right - left + 1);
    }
    return maxLen;
}
// Time Complexity: O(N) | Space Complexity: O(1)
```"""
    },
    "17": {
        "python": """```python
def total_fruit(self, fruits: list[int]) -> int:
    from collections import defaultdict
    count = defaultdict(int)
    left = max_val = 0
    for right in range(len(fruits)):
        count[fruits[right]] += 1
        while len(count) > 2:
            count[fruits[left]] -= 1
            if count[fruits[left]] == 0:
                del count[fruits[left]]
            left += 1
        max_val = max(max_val, right - left + 1)
    return max_val
# Time Complexity: O(N) | Space Complexity: O(1)
```""",
        "csharp": """```csharp
public int TotalFruit(int[] fruits) {
    Dictionary<int, int> count = new Dictionary<int, int>();
    int left = 0, max = 0;
    for (int right = 0; right < fruits.Length; right++) {
        count[fruits[right]] = count.GetValueOrDefault(fruits[right], 0) + 1;
        while (count.Count > 2) {
            count[fruits[left]]--;
            if (count[fruits[left]] == 0) count.Remove(fruits[left]);
            left++;
        }
        max = Math.Max(max, right - left + 1);
    }
    return max;
}
// Time Complexity: O(N) | Space Complexity: O(1)
```"""
    }
}

base_dir = r"C:\Users\hari\Documents\DBA\books\spec_driven_interviews\chapters\12-hashmaps-sliding-windows\snippets"
for k, v in translations.items():
    with open(os.path.join(base_dir, "python", f"code_block_{k}.md"), "w", encoding="utf-8") as f:
        f.write(v["python"])
    with open(os.path.join(base_dir, "csharp", f"code_block_{k}.md"), "w", encoding="utf-8") as f:
        f.write(v["csharp"])

print("Translations for chapter 12 partly applied!")
