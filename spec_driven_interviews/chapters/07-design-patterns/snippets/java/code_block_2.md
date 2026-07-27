```java
// Example of a fluent, type-safe builder for transactions
TransactionRecord tx = new TransactionRecordBuilder()
    .withId(UUID.randomUUID())
    .fromAccount(sourceId)
    .toAccount(destId)
    .withAmount(new BigDecimal("100.00"))
    .inCurrency("USD")
    .atTimestamp(Instant.now())
    .build(); // Immutability and invariants are validated in build()
```
