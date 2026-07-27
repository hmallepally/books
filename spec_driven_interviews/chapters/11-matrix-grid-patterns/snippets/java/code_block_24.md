```java
public boolean solution(String cell1, String cell2) {
  int sum1 = (cell1.charAt(0) - 'A') + (cell1.charAt(1) - '1');
  int sum2 = (cell2.charAt(0) - 'A') + (cell2.charAt(1) - '1');
  return (sum1 % 2) == (sum2 % 2);
}
```
