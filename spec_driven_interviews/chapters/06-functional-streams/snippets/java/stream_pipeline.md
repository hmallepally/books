```java
transactions.stream()
    .filter(t -> t.amount() > 100)
    .peek(t -> log.debug("Passed Filter: {}", t.id()))
    .map(Transaction::merchantId)
    .collect(Collectors.toList());
```