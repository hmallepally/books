```java
// Imperative anti-pattern: Hard to read, mutable state, difficult to parallelize
Map<UUID, BigDecimal> volumes = new HashMap<>();
for (TransactionRecord tx : transactions) {
    if (tx.amount().compareTo(threshold) >= 0) {
        UUID merchantId = tx.destinationAccountId();
        BigDecimal currentSum = volumes.getOrDefault(merchantId, BigDecimal.ZERO);
        volumes.put(merchantId, currentSum.add(tx.amount()));
    }
}
```
