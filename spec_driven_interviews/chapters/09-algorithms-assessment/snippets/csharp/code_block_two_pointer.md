```csharp
namespace AuraPay.Algorithms;

/// <summary>
/// Implements the Two-Pointer pattern to find two numbers that sum to a target
/// in a 1-indexed sorted array.
/// Time Complexity: O(N) where N is the size of the array.
/// Space Complexity: O(1) auxiliary space.
/// </summary>
public class TwoPointerSolver
{
    /// <summary>
    /// Finds indices of the two numbers that add up to the target.
    /// Uses two pointers moving from opposite ends inward.
    /// </summary>
    public int[] FindMatchingNumbers(int[] numbers, int target)
    {
        int start = 1;
        int last = numbers.Length;
        int[] output = new int[2];

        while (start < last)
        {
            int sum = numbers[start - 1] + numbers[last - 1];
            if (sum == target)
            {
                output[0] = start;
                output[1] = last;
                return output;
            }
            if (sum < target)
            {
                start++;
            }
            else
            {
                last--;
            }
        }
        return output; // Returns [0, 0] if no match is found
    }
}
```
