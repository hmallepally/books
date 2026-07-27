```csharp
namespace AuraPay.Algorithms;

/// <summary>
/// Sorts an array containing numbers from 1 to N in-place.
/// Time Complexity: O(N) where N is the size of the array.
/// Space Complexity: O(1) auxiliary space.
/// </summary>
public class CyclicSort
{
    public void Sort(int[] nums)
    {
        int i = 0;
        while (i < nums.Length)
        {
            int correctIndex = nums[i] - 1; // Value X belongs at index X-1
            if (nums[i] != nums[correctIndex])
            {
                Swap(nums, i, correctIndex); // Swap to correct position
            }
            else
            {
                i++; // Increment only when correct
            }
        }
    }

    private void Swap(int[] nums, int i, int j)
    {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
}
```
