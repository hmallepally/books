```java
package com.aurapay.domain;

import java.math.BigDecimal;
import java.time.Instant;
import java.util.Objects;
import java.util.UUID;

/**
 * Represents an immutable, validated financial transaction record in AuraPay.
 * Enforces pre-conditions on initialization.
 */
public record TransactionRecord(
    UUID transactionId,
    UUID sourceAccountId,
    UUID destinationAccountId,
    BigDecimal amount,
    String currency,
    Instant timestamp
) {
    public TransactionRecord {
        Objects.requireNonNull(transactionId, "Transaction ID cannot be null");
        Objects.requireNonNull(sourceAccountId, "Source Account ID cannot be null");
        Objects.requireNonNull(destinationAccountId, "Destination Account ID cannot be null");
        Objects.requireNonNull(amount, "Amount cannot be null");
        Objects.requireNonNull(currency, "Currency cannot be null");
        Objects.requireNonNull(timestamp, "Timestamp cannot be null");

        if (sourceAccountId.equals(destinationAccountId)) {
            throw new IllegalArgumentException("Source and destination accounts must be distinct");
        }
        if (amount.compareTo(BigDecimal.ZERO) <= 0) {
            throw new IllegalArgumentException("Transaction amount must be strictly positive");
        }
        if (currency.trim().isEmpty()) {
            throw new IllegalArgumentException("Currency code cannot be empty");
        }
    }
}
```
