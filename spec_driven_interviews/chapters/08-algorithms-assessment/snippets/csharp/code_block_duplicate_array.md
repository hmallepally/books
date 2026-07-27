```csharp
namespace AuraPay.Algorithms;

/// <summary>
/// Finds the duplicate number in an array using Floyd's Cycle Detection.
/// Time Complexity: O(N) where N is the size of the array.
/// Space Complexity: O(1) auxiliary space.
/// Constraint: The array must contain N + 1 elements, each between 1 and N.
/// </summary>
public class DuplicateArrayFinder
{
    public int FindDuplicate(int[] nums)
    {
        // Phase 1: Detect cycle (meeting point)
        int slow = nums[0];
        int fast = nums[0];

        do
        {
            slow = nums[slow];          // Move 1 step
            fast = nums[nums[fast]];    // Move 2 steps
        } while (slow != fast);

        // Phase 2: Find cycle entrance (duplicate value)
        slow = nums[0]; // Reset slow to start
        while (slow != fast)
        {
            slow = nums[slow]; // Move 1 step
            fast = nums[fast]; // Move 1 step
        }

        return slow; // The duplicate value
    }
}
```
