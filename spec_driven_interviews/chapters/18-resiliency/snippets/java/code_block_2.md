```java
// Anti-pattern: Dual-Write
public void completeTransaction(TransactionRecord tx) {
    database.save(tx); // Database Write
    kafkaTemplate.send("transaction-topic", tx); // Network Call
}
```
