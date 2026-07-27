```java
package com.aurapay.analytics;

import com.aurapay.domain.TransactionRecord;
import java.math.BigDecimal;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.UUID;
import java.util.stream.Collectors;

/**
 * Demonstrates high-performance batch transaction analytics using Java Streams.
 */
public class TransactionAnalytics {

    /**
     * Processes a list of transactions to aggregate total volume per merchant,
     * filtering out high-risk or low-value records.
     */
    public Map<UUID, BigDecimal> aggregateMerchantVolumes(
        List<TransactionRecord> transactions, 
        BigDecimal minAmountThreshold
    ) {
        Objects.requireNonNull(transactions, "Transaction list cannot be null");
        Objects.requireNonNull(minAmountThreshold, "Threshold cannot be null");

        // Declarative functional pipeline
        return transactions.stream()
            // 1. Filter: Retain only transactions meeting the value criteria (side-effect-free)
            .filter(t -> t.amount().compareTo(minAmountThreshold) >= 0)
            
            // 2. Collect: Group by merchant and sum the transaction volume
            .collect(Collectors.toMap(
                TransactionRecord::destinationAccountId, // Key mapper: Merchant ID
                TransactionRecord::amount,              // Value mapper: Transaction amount
                BigDecimal::add                         // Merge function: Sum volumes
            ));
    }

    /**
     * Finds the transaction IDs of all transfers exceeding a safety limit, 
     * sorted chronologically.
     */
    public List<UUID> getHighValueTransactionIds(
        List<TransactionRecord> transactions, 
        BigDecimal limit
    ) {
        return transactions.stream()
            .filter(t -> t.amount().compareTo(limit) > 0)
            .map(TransactionRecord::transactionId)
            .collect(Collectors.toList());
    }
}
```
