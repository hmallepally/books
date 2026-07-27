```csharp
using System;
using System.Collections.Generic;

namespace AuraPay.Algorithms
{
    /// <summary>
    /// Implements the Sliding Window Maximum algorithm using a Monotonic Deque.
    /// </summary>
    public class SlidingWindowSolver
    {
        public int[] MaxSlidingWindow(int[] nums, int k)
        {
            if (nums == null || nums.Length == 0 || k <= 0)
            {
                return new int[0];
            }

            int n = nums.Length;
            int[] result = new int[n - k + 1];
            int ri = 0;

            // In C#, we can use LinkedList<int> as a double-ended queue (deque)
            LinkedList<int> q = new LinkedList<int>();

            for (int i = 0; i < n; i++)
            {
                // 1. Remove indices that are out of the current window boundary
                if (q.Count > 0 && q.First.Value < i - k + 1)
                {
                    q.RemoveFirst();
                }

                // 2. Maintain monotonic invariant: Remove indices of elements smaller
                // than the current element from the tail of the deque
                while (q.Count > 0 && nums[q.Last.Value] < nums[i])
                {
                    q.RemoveLast();
                }

                // 3. Add current element's index to the tail
                q.AddLast(i);

                // 4. If window size has reached K, store the maximum in the result
                if (i >= k - 1)
                {
                    result[ri++] = nums[q.First.Value];
                }
            }

            return result;
        }
    }
}
```
