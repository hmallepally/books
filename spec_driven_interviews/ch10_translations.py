import os

translations = {
    "1": {
        "python": """```python
# Retains elements satisfying a condition, overwrites list in-place
write = 0
for read in range(len(arr)):
    if keep_condition(arr[read]):
        arr[write] = arr[read]
        write += 1
# Result is arr[0..write-1], return write as the new length
```""",
        "csharp": """```csharp
// Retains elements satisfying a condition, overwrites array in-place
int write = 0;
for (int read = 0; read < arr.Length; read++) {
    if (KeepCondition(arr[read])) {
        arr[write] = arr[read];
        write++;
    }
}
// Result is arr[0..write-1], return write as the new length
```"""
    },
    "2": {
        "python": """```python
left, right = 0, len(arr) - 1
while left < right:
    # Process or compare arr[left] and arr[right]
    # Optionally skip invalid elements
    left += 1
    right -= 1
```""",
        "csharp": """```csharp
int left = 0, right = arr.Length - 1;
while (left < right) {
    // Process or compare arr[left] and arr[right]
    // Optionally skip invalid elements
    left++;
    right--;
}
```"""
    },
    "3": {
        "python": """```python
def first_uniq_char(self, s: str) -> int:
    if not s:
        return -1

    # Pass 1: Count frequency of each character
    counts = [0] * 256
    for char in s:
        counts[ord(char)] += 1

    # Pass 2: Find first character with frequency exactly 1
    for i, char in enumerate(s):
        if counts[ord(char)] == 1:
            return i

    return -1 # All characters repeat
# Time: O(N), Space: O(1) — the counts list is constant size
```""",
        "csharp": """```csharp
public int FirstUniqChar(string s) {
    if (string.IsNullOrEmpty(s)) return -1;

    // Pass 1: Count frequency of each character
    int[] counts = new int[256];
    foreach (char c in s) {
        counts[c]++;
    }

    // Pass 2: Find first character with frequency exactly 1
    for (int i = 0; i < s.Length; i++) {
        if (counts[s[i]] == 1) return i;
    }

    return -1; // All characters repeat
}
// Time: O(N), Space: O(1) — the int[256] is constant size
```"""
    },
    "4": {
        "python": """```python
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
```""",
        "csharp": """```csharp
public int Compress(char[] chars) {
    if (chars == null || chars.Length == 0) return 0;

    int write = 0; // Write pointer for compressed output
    int read = 0;  // Read pointer scanning input

    while (read < chars.Length) {
        char current = chars[read];
        int count = 0;

        // Count consecutive occurrences of current character
        while (read < chars.Length && chars[read] == current) {
            read++;
            count++;
        }

        // Write the character itself
        chars[write++] = current;

        // Write the count digits (only if count > 1)
        if (count > 1) {
            // Convert count to individual digit characters
            foreach (char digit in count.ToString()) {
                chars[write++] = digit;
            }
        }
    }

    return write;
}
// Time: O(N), Space: O(1) auxiliary
```"""
    },
    "5": {
        "python": """```python
def is_palindrome(self, s: str) -> bool:
    if s is None:
        return False

    left, right = 0, len(s) - 1

    while left < right:
        # Skip non-alphanumeric from the left
        while left < right and not s[left].isalnum():
            left += 1
        # Skip non-alphanumeric from the right
        while left < right and not s[right].isalnum():
            right -= 1

        # Compare characters (case-insensitive)
        if s[left].lower() != s[right].lower():
            return False

        left += 1
        right -= 1

    return True
# Time: O(N), Space: O(1)
```""",
        "csharp": """```csharp
public bool IsPalindrome(string s) {
    if (s == null) return false;

    int left = 0, right = s.Length - 1;

    while (left < right) {
        // Skip non-alphanumeric from the left
        while (left < right && !char.IsLetterOrDigit(s[left])) {
            left++;
        }
        // Skip non-alphanumeric from the right
        while (left < right && !char.IsLetterOrDigit(s[right])) {
            right--;
        }

        // Compare characters (case-insensitive)
        if (char.ToLower(s[left]) != char.ToLower(s[right])) {
            return false;
        }

        left++;
        right--;
    }

    return true;
}
// Time: O(N), Space: O(1)
```"""
    },
    "6": {
        "python": """```python
def move_zeroes(self, nums: list[int]) -> None:
    if not nums:
        return

    # Pass 1: Copy all non-zero elements to the front
    write = 0
    for read in range(len(nums)):
        if nums[read] != 0:
            nums[write] = nums[read]
            write += 1

    # Pass 2: Fill remaining positions with zeros
    while write < len(nums):
        nums[write] = 0
        write += 1
# Time: O(N), Space: O(1)
```""",
        "csharp": """```csharp
public void MoveZeroes(int[] nums) {
    if (nums == null || nums.Length == 0) return;

    // Pass 1: Copy all non-zero elements to the front
    int write = 0;
    for (int read = 0; read < nums.Length; read++) {
        if (nums[read] != 0) {
            nums[write++] = nums[read];
        }
    }

    // Pass 2: Fill remaining positions with zeros
    while (write < nums.Length) {
        nums[write++] = 0;
    }
}
// Time: O(N), Space: O(1)
```"""
    },
    "7": {
        "python": """```python
def remove_duplicates(self, nums: list[int]) -> int:
    if not nums:
        return 0

    write = 1 # First element is always unique
    for read in range(1, len(nums)):
        if nums[read] != nums[write - 1]:
            nums[write] = nums[read]
            write += 1

    return write
# Time: O(N), Space: O(1)
```""",
        "csharp": """```csharp
public int RemoveDuplicates(int[] nums) {
    if (nums == null || nums.Length == 0) return 0;

    int write = 1; // First element is always unique
    for (int read = 1; read < nums.Length; read++) {
        if (nums[read] != nums[write - 1]) {
            nums[write++] = nums[read];
        }
    }

    return write;
}
// Time: O(N), Space: O(1)
```"""
    },
    "8": {
        "python": """```python
def single_number(self, nums: list[int]) -> int:
    result = 0
    for num in nums:
        result ^= num # Pairs cancel, unique value survives
    return result
# Time: O(N), Space: O(1)
```""",
        "csharp": """```csharp
public int SingleNumber(int[] nums) {
    int result = 0;
    foreach (int num in nums) {
        result ^= num; // Pairs cancel, unique value survives
    }
    return result;
}
// Time: O(N), Space: O(1)
```"""
    },
    "9": {
        "python": """```python
def is_valid(self, s: str) -> bool:
    if not s or len(s) % 2 != 0:
        return False

    stack = []

    for c in s:
        if c == '(': stack.append(')')
        elif c == '{': stack.append('}')
        elif c == '[': stack.append(']')
        else:
            if not stack or stack.pop() != c:
                return False

    return len(stack) == 0 # Stack must be empty
# Time: O(N), Space: O(N) worst case for the stack
```""",
        "csharp": """```csharp
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
```"""
    },
    "10": {
        "python": """```python
def reverse_string(self, s: list[str]) -> None:
    if not s or len(s) <= 1:
        return

    left, right = 0, len(s) - 1
    while left < right:
        s[left], s[right] = s[right], s[left]
        left += 1
        right -= 1
# Time: O(N), Space: O(1)
```""",
        "csharp": """```csharp
public void ReverseString(char[] s) {
    if (s == null || s.Length <= 1) return;

    int left = 0, right = s.Length - 1;
    while (left < right) {
        char temp = s[left];
        s[left] = s[right];
        s[right] = temp;
        left++;
        right--;
    }
}
// Time: O(N), Space: O(1)
```"""
    },
    "11": {
        "python": """```python
def pivot_index(self, nums: list[int]) -> int:
    if not nums:
        return -1

    total_sum = sum(nums)
    left_sum = 0
    
    for i, num in enumerate(nums):
        # right_sum = total_sum - left_sum - num
        if left_sum == total_sum - left_sum - num:
            return i
        left_sum += num

    return -1
# Time: O(N), Space: O(1)
```""",
        "csharp": """```csharp
public int PivotIndex(int[] nums) {
    if (nums == null) return -1;

    int totalSum = 0;
    foreach (int num in nums) totalSum += num;

    int leftSum = 0;
    for (int i = 0; i < nums.Length; i++) {
        // rightSum = totalSum - leftSum - nums[i]
        if (leftSum == totalSum - leftSum - nums[i]) return i;
        leftSum += nums[i];
    }

    return -1;
}
// Time: O(N), Space: O(1)
```"""
    },
    "12": {
        "python": """```python
def is_monotonic(self, nums: list[int]) -> bool:
    if not nums or len(nums) <= 2:
        return True

    increasing = True
    decreasing = True

    for i in range(len(nums) - 1):
        if nums[i] > nums[i + 1]: increasing = False
        if nums[i] < nums[i + 1]: decreasing = False

    return increasing or decreasing
# Time: O(N), Space: O(1)
```""",
        "csharp": """```csharp
public bool IsMonotonic(int[] nums) {
    if (nums == null || nums.Length <= 2) return true;

    bool increasing = true;
    bool decreasing = true;

    for (int i = 0; i < nums.Length - 1; i++) {
        if (nums[i] > nums[i + 1]) increasing = false;
        if (nums[i] < nums[i + 1]) decreasing = false;
    }

    return increasing || decreasing;
}
// Time: O(N), Space: O(1)
```"""
    },
    "13": {
        "python": """```python
def neighbor_sum(self, a: list[int]) -> list[int]:
    if not a:
        return []
    n = len(a)
    b = [0] * n

    for i in range(n):
        left_val = a[i - 1] if i > 0 else 0
        right_val = a[i + 1] if i < n - 1 else 0
        b[i] = left_val + a[i] + right_val

    return b
# Time: O(N), Space: O(N) for output array
```""",
        "csharp": """```csharp
public int[] NeighborSum(int[] a) {
    if (a == null) return new int[0];
    int n = a.Length;
    int[] b = new int[n];

    for (int i = 0; i < n; i++) {
        int leftVal  = (i > 0) ? a[i - 1] : 0;
        int rightVal = (i < n - 1) ? a[i + 1] : 0;
        b[i] = leftVal + a[i] + rightVal;
    }

    return b;
}
// Time: O(N), Space: O(N) for output array
```"""
    },
    "14": {
        "python": """```python
def max_sum_subarray(self, nums: list[int], k: int) -> int:
    if not nums or len(nums) < k or k <= 0:
        return 0

    # Initialize sum of first window
    window_sum = sum(nums[:k])
    max_sum = window_sum

    # Slide the window: add right element, remove left element
    for i in range(k, len(nums)):
        window_sum += nums[i] - nums[i - k]
        max_sum = max(max_sum, window_sum)

    return max_sum
# Time: O(N), Space: O(1)
```""",
        "csharp": """```csharp
public int MaxSumSubarray(int[] nums, int k) {
    if (nums == null || nums.Length < k || k <= 0) return 0;

    // Initialize sum of first window
    int windowSum = 0;
    for (int i = 0; i < k; i++) windowSum += nums[i];

    int maxSum = windowSum;

    // Slide the window: add right element, remove left element
    for (int i = k; i < nums.Length; i++) {
        windowSum += nums[i] - nums[i - k];
        maxSum = Math.Max(maxSum, windowSum);
    }

    return maxSum;
}
// Time: O(N), Space: O(1)
```"""
    },
    "15": {
        "python": """```python
def find_the_difference(self, s: str, t: str) -> str:
    result = 0
    for c in s: result ^= ord(c)
    for c in t: result ^= ord(c)
    return chr(result) # Only the unpaired character survives
# Time: O(N), Space: O(1)
```""",
        "csharp": """```csharp
public char FindTheDifference(string s, string t) {
    char result = (char)0;
    foreach (char c in s) result ^= c;
    foreach (char c in t) result ^= c;
    return result; // Only the unpaired character survives
}
// Time: O(N), Space: O(1)
```"""
    },
    "16": {
        "python": """```python
def transform_words(self, words: list[str]) -> list[str]:
    if not words:
        return []
    result = [""] * len(words)

    for i in range(len(words)):
        if len(words[i]) % 2 != 0:
            result[i] = words[i].upper()
        else:
            result[i] = words[i][::-1]

    return result
# Time: O(N * K) where K is average word length, Space: O(N * K) for output
```""",
        "csharp": """```csharp
public string[] TransformWords(string[] words) {
    if (words == null) return new string[0];
    string[] result = new string[words.Length];

    for (int i = 0; i < words.Length; i++) {
        if (words[i].Length % 2 != 0) {
            result[i] = words[i].ToUpper();
        } else {
            char[] arr = words[i].ToCharArray();
            Array.Reverse(arr);
            result[i] = new string(arr);
        }
    }

    return result;
}
// Time: O(N * K) where K is average word length, Space: O(N * K) for output
```"""
    },
    "17": {
        "python": """```python
def are_occurrences_equal(self, s: str) -> bool:
    if not s:
        return True

    from collections import Counter
    counts = Counter(s)
    
    expected = 0
    for count in counts.values():
        if count > 0:
            if expected == 0: expected = count
            elif count != expected: return False

    return True
# Time: O(N), Space: O(1)
```""",
        "csharp": """```csharp
public bool AreOccurrencesEqual(string s) {
    if (string.IsNullOrEmpty(s)) return true;

    int[] counts = new int[128];
    foreach (char c in s) counts[c]++;

    int expected = 0;
    foreach (int count in counts) {
        if (count > 0) {
            if (expected == 0) expected = count;
            else if (count != expected) return false;
        }
    }

    return true;
}
// Time: O(N), Space: O(1)
```"""
    },
    "18": {
        "python": """```python
def remove_element(self, nums: list[int], val: int) -> int:
    if nums is None:
        return 0

    write = 0
    for read in range(len(nums)):
        if nums[read] != val:
            nums[write] = nums[read]
            write += 1

    return write
# Time: O(N), Space: O(1)
```""",
        "csharp": """```csharp
public int RemoveElement(int[] nums, int val) {
    if (nums == null) return 0;

    int write = 0;
    for (int read = 0; read < nums.Length; read++) {
        if (nums[read] != val) {
            nums[write++] = nums[read];
        }
    }

    return write;
}
// Time: O(N), Space: O(1)
```"""
    },
    "19": {
        "python": """```python
def is_alternating_parity(self, nums: list[int]) -> bool:
    if not nums or len(nums) <= 1:
        return True

    for i in range(len(nums) - 1):
        if (abs(nums[i]) % 2) == (abs(nums[i + 1]) % 2):
            return False

    return True
# Time: O(N), Space: O(1)
```""",
        "csharp": """```csharp
public bool IsAlternatingParity(int[] nums) {
    if (nums == null || nums.Length <= 1) return true;

    for (int i = 0; i < nums.Length - 1; i++) {
        // Use Math.Abs for safety with negative numbers
        if (Math.Abs(nums[i] % 2) == Math.Abs(nums[i + 1] % 2)) {
            return false;
        }
    }

    return true;
}
// Time: O(N), Space: O(1)
```"""
    },
    "20": {
        "python": """```python
def two_sum(self, nums: list[int], target: int) -> list[int]:
    seen = {}

    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i

    return [] # Should not reach here per problem guarantee
# Time: O(N), Space: O(N)
```""",
        "csharp": """```csharp
public int[] TwoSum(int[] nums, int target) {
    Dictionary<int, int> seen = new Dictionary<int, int>();

    for (int i = 0; i < nums.Length; i++) {
        int complement = target - nums[i];
        if (seen.ContainsKey(complement)) {
            return new int[]{seen[complement], i};
        }
        seen[nums[i]] = i;
    }

    return new int[]{}; // Should not reach here per problem guarantee
}
// Time: O(N), Space: O(N)
```"""
    },
    "21": {
        "python": """```python
def majority_element(self, nums: list[int]) -> int:
    candidate = nums[0]
    count = 1

    for i in range(1, len(nums)):
        if count == 0:
            candidate = nums[i]
            count = 1
        elif nums[i] == candidate:
            count += 1
        else:
            count -= 1

    return candidate
# Time: O(N), Space: O(1)
```""",
        "csharp": """```csharp
public int MajorityElement(int[] nums) {
    int candidate = nums[0];
    int count = 1;

    for (int i = 1; i < nums.Length; i++) {
        if (count == 0) {
            candidate = nums[i];
            count = 1;
        } else if (nums[i] == candidate) {
            count++;
        } else {
            count--;
        }
    }

    return candidate;
}
// Time: O(N), Space: O(1)
```"""
    },
    "22": {
        "python": """```python
def plus_one(self, digits: list[int]) -> list[int]:
    for i in range(len(digits) - 1, -1, -1):
        digits[i] += 1
        if digits[i] < 10:
            return digits # No further carry needed
        digits[i] = 0 # Carry to next position

    # All digits were 9 — need a new array [1, 0, 0, ..., 0]
    return [1] + [0] * len(digits)
# Time: O(N), Space: O(1) amortized (O(N) only for all-9s edge case)
```""",
        "csharp": """```csharp
public int[] PlusOne(int[] digits) {
    for (int i = digits.Length - 1; i >= 0; i--) {
        digits[i]++;
        if (digits[i] < 10) {
            return digits; // No further carry needed
        }
        digits[i] = 0; // Carry to next position
    }

    // All digits were 9 — need a new array [1, 0, 0, ..., 0]
    int[] result = new int[digits.Length + 1];
    result[0] = 1;
    return result;
}
// Time: O(N), Space: O(1) amortized (O(N) only for all-9s edge case)
```"""
    },
    "23": {
        "python": """```python
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
```""",
        "csharp": """```csharp
public int AdjacentElementsProduct(int[] inputArray) {
    if (inputArray == null || inputArray.Length < 2) return 0;

    int maxProd = inputArray[0] * inputArray[1];

    for (int i = 1; i < inputArray.Length - 1; i++) {
        int prod = inputArray[i] * inputArray[i + 1];
        if (prod > maxProd) {
            maxProd = prod;
        }
    }

    return maxProd;
}
// Time: O(N), Space: O(1)
```"""
    },
    "24": {
        "python": """```python
def century_from_year(self, year: int) -> int:
    return (year + 99) // 100
# Time: O(1), Space: O(1)
```""",
        "csharp": """```csharp
public int CenturyFromYear(int year) {
    return (year + 99) / 100;
}
// Time: O(1), Space: O(1)
```"""
    },
    "25": {
        "python": """```python
def all_longest_strings(self, input_array: list[str]) -> list[str]:
    # Pass 1: Find the maximum length
    max_length = 0
    for s in input_array:
        if len(s) > max_length:
            max_length = len(s)

    # Pass 2: Collect strings matching the max length
    result = []
    for s in input_array:
        if len(s) == max_length:
            result.append(s)

    return result
# Time: O(N), Space: O(N) for output
```""",
        "csharp": """```csharp
public string[] AllLongestStrings(string[] inputArray) {
    // Pass 1: Find the maximum length
    int maxLength = 0;
    foreach (string s in inputArray) {
        if (s.Length > maxLength) {
            maxLength = s.Length;
        }
    }

    // Pass 2: Collect strings matching the max length
    List<string> result = new List<string>();
    foreach (string s in inputArray) {
        if (s.Length == maxLength) {
            result.Add(s);
        }
    }

    return result.ToArray();
}
// Time: O(N), Space: O(N) for output
```"""
    },
    "26": {
        "python": """```python
def common_character_count(self, s1: str, s2: str) -> int:
    count1 = [0] * 26
    count2 = [0] * 26

    for c in s1: count1[ord(c) - ord('a')] += 1
    for c in s2: count2[ord(c) - ord('a')] += 1

    common = 0
    for i in range(26):
        common += min(count1[i], count2[i])

    return common
# Time: O(N + M), Space: O(1) — fixed 26-element lists
```""",
        "csharp": """```csharp
public int CommonCharacterCount(string s1, string s2) {
    int[] count1 = new int[26];
    int[] count2 = new int[26];

    foreach (char c in s1) count1[c - 'a']++;
    foreach (char c in s2) count2[c - 'a']++;

    int common = 0;
    for (int i = 0; i < 26; i++) {
        common += Math.Min(count1[i], count2[i]);
    }

    return common;
}
// Time: O(N + M), Space: O(1) — fixed 26-element arrays
```"""
    },
    "27": {
        "python": """```python
def is_lucky(self, n: int) -> bool:
    s = str(n)
    mid = len(s) // 2
    sum1 = 0
    sum2 = 0

    for i in range(mid):
        sum1 += int(s[i])       # First half digit
        sum2 += int(s[i + mid]) # Second half digit

    return sum1 == sum2
# Time: O(D) where D is digit count, Space: O(D) for string conversion
```""",
        "csharp": """```csharp
public bool IsLucky(int n) {
    string s = n.ToString();
    int mid = s.Length / 2;
    int sum1 = 0, sum2 = 0;

    for (int i = 0; i < mid; i++) {
        sum1 += s[i] - '0';       // First half digit
        sum2 += s[i + mid] - '0'; // Second half digit
    }

    return sum1 == sum2;
}
// Time: O(D) where D is digit count, Space: O(D) for string conversion
```"""
    },
    "28": {
        "python": """```python
def sort_by_height(self, a: list[int]) -> list[int]:
    # Step 1: Extract all non-tree heights
    heights = [h for h in a if h != -1]

    # Step 2: Sort the extracted heights
    heights.sort()

    # Step 3: Reinsert sorted heights at non-tree positions
    index = 0
    for i in range(len(a)):
        if a[i] != -1:
            a[i] = heights[index]
            index += 1

    return a
# Time: O(N log N) for sorting, Space: O(N) for extracted list
```""",
        "csharp": """```csharp
public int[] SortByHeight(int[] a) {
    // Step 1: Extract all non-tree heights
    List<int> heights = new List<int>();
    foreach (int h in a) {
        if (h != -1) heights.Add(h);
    }

    // Step 2: Sort the extracted heights
    heights.Sort();

    // Step 3: Reinsert sorted heights at non-tree positions
    int index = 0;
    for (int i = 0; i < a.Length; i++) {
        if (a[i] != -1) {
            a[i] = heights[index++];
        }
    }

    return a;
}
// Time: O(N log N) for sorting, Space: O(N) for extracted list
```"""
    },
    "29": {
        "python": """```python
def alternating_sums(self, a: list[int]) -> list[int]:
    team1 = 0
    team2 = 0

    for i in range(len(a)):
        if i % 2 == 0:
            team1 += a[i]
        else:
            team2 += a[i]

    return [team1, team2]
# Time: O(N), Space: O(1)
```""",
        "csharp": """```csharp
public int[] AlternatingSums(int[] a) {
    int team1 = 0, team2 = 0;

    for (int i = 0; i < a.Length; i++) {
        if (i % 2 == 0) {
            team1 += a[i];
        } else {
            team2 += a[i];
        }
    }

    return new int[]{team1, team2};
}
// Time: O(N), Space: O(1)
```"""
    },
    "30": {
        "python": """```python
def add_border(self, picture: list[str]) -> list[str]:
    new_width = len(picture[0]) + 2
    result = [""] * (len(picture) + 2)

    # Build the border row
    border = '*' * new_width

    # Top border
    result[0] = border

    # Wrap each interior row with side asterisks
    for i in range(len(picture)):
        result[i + 1] = f"*{picture[i]}*"

    # Bottom border
    result[-1] = border

    return result
# Time: O(rows * cols), Space: O(rows * cols) for output
```""",
        "csharp": """```csharp
public string[] AddBorder(string[] picture) {
    int newWidth = picture[0].Length + 2;
    string[] result = new string[picture.Length + 2];

    // Build the border row
    string border = new string('*', newWidth);

    // Top border
    result[0] = border;

    // Wrap each interior row with side asterisks
    for (int i = 0; i < picture.Length; i++) {
        result[i + 1] = "*" + picture[i] + "*";
    }

    // Bottom border
    result[result.Length - 1] = border;

    return result;
}
// Time: O(rows * cols), Space: O(rows * cols) for output
```"""
    },
    "31": {
        "python": """```python
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
```""",
        "csharp": """```csharp
public int ArrayChange(int[] inputArray) {
    int moves = 0;

    for (int i = 1; i < inputArray.Length; i++) {
        if (inputArray[i] <= inputArray[i - 1]) {
            // Calculate the minimum increment needed
            int deficit = inputArray[i - 1] - inputArray[i] + 1;
            inputArray[i] += deficit;
            moves += deficit;
        }
    }

    return moves;
}
// Time: O(N), Space: O(1)
```"""
    },
    "32": {
        "python": """```python
def matrix_elements_sum(self, matrix: list[list[int]]) -> int:
    rows = len(matrix)
    cols = len(matrix[0])
    total = 0

    for c in range(cols):
        for r in range(rows):
            if matrix[r][c] == 0:
                break # All rooms below are haunted — skip rest of column
            total += matrix[r][c]

    return total
# Time: O(rows * cols), Space: O(1)
```""",
        "csharp": """```csharp
public int MatrixElementsSum(int[][] matrix) {
    int rows = matrix.Length;
    int cols = matrix[0].Length;
    int total = 0;

    for (int c = 0; c < cols; c++) {
        for (int r = 0; r < rows; r++) {
            if (matrix[r][c] == 0) {
                break; // All rooms below are haunted — skip rest of column
            }
            total += matrix[r][c];
        }
    }

    return total;
}
// Time: O(rows * cols), Space: O(1)
```"""
    },
    "33": {
        "python": """```python
def almost_increasing_sequence(self, sequence: list[int]) -> bool:
    count = 0   # Number of violations
    bad_idx = -1  # Index of first violation

    for i in range(len(sequence) - 1):
        if sequence[i] >= sequence[i + 1]:
            count += 1
            bad_idx = i
            if count > 1: return False # More than one violation

    if count == 0: return True # Already strictly increasing

    # Try removing element at bad_idx
    if bad_idx == 0 or sequence[bad_idx - 1] < sequence[bad_idx + 1]:
        return True

    # Try removing element at bad_idx + 1
    if bad_idx + 2 >= len(sequence) or sequence[bad_idx] < sequence[bad_idx + 2]:
        return True

    return False
# Time: O(N), Space: O(1)
```""",
        "csharp": """```csharp
public bool AlmostIncreasingSequence(int[] sequence) {
    int count = 0;   // Number of violations
    int badIdx = -1;  // Index of first violation

    for (int i = 0; i < sequence.Length - 1; i++) {
        if (sequence[i] >= sequence[i + 1]) {
            count++;
            badIdx = i;
            if (count > 1) return false; // More than one violation
        }
    }

    if (count == 0) return true; // Already strictly increasing

    // Try removing element at badIdx
    if (badIdx == 0 || sequence[badIdx - 1] < sequence[badIdx + 1]) {
        return true;
    }

    // Try removing element at badIdx + 1
    if (badIdx + 2 >= sequence.Length || sequence[badIdx] < sequence[badIdx + 2]) {
        return true;
    }

    return false;
}
// Time: O(N), Space: O(1)
```"""
    },
    "34": {
        "python": """```python
def reverse_in_parentheses(self, s: str) -> str:
    stack = [[]]

    for c in s:
        if c == '(':
            stack.append([]) # Start new nested context
        elif c == ')':
            inner = stack.pop()  # Pop innermost context
            inner.reverse()       # Reverse it
            stack[-1].extend(inner) # Append to enclosing context
        else:
            stack[-1].append(c)  # Accumulate character

    return "".join(stack[0])
# Time: O(N^2) worst case for nested reversals, Space: O(N)
```""",
        "csharp": """```csharp
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
```"""
    }
}

base_dir = r"C:\Users\hari\Documents\DBA\books\spec_driven_interviews\chapters\10-implementation-patterns\snippets"
for k, v in translations.items():
    with open(os.path.join(base_dir, "python", f"code_block_{k}.md"), "w", encoding="utf-8") as f:
        f.write(v["python"])
    with open(os.path.join(base_dir, "csharp", f"code_block_{k}.md"), "w", encoding="utf-8") as f:
        f.write(v["csharp"])

print("Translations for chapter 10 applied!")
