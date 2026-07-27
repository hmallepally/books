```java
public int[] alternatingSums(int[] a) {
    int team1 = 0, team2 = 0;

    for (int i = 0; i < a.length; i++) {
        if (i % 2 == 0) {
            team1 += a[i];
        } else {
            team2 += a[i];
        }
    }

    return new int[]{team1, team2};
}
// Time: O(N), Space: O(1)
```
