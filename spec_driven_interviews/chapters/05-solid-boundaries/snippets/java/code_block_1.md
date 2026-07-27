```java
package com.aurapay.processing;

import com.aurapay.domain.LedgerAccount;
import com.aurapay.domain.TransactionRecord;
import java.math.BigDecimal;
import java.util.Objects;
import java.util.UUID;

/**
 * Abstraction for database operations (Dependency Inversion Principle).
 */
public interface LedgerRepository {
    LedgerAccount findById(UUID accountId);
    void save(LedgerAccount account);
}

/**
 * Abstraction for fee calculations (Open/Closed Principle).
 */
public interface FeeCalculator {
    BigDecimal calculate(TransactionRecord transaction);
}

/**
 * Interface Segregation Principle: Focused notification dispatch interface.
 */
public interface TransactionNotificationSender {
    void sendNotification(TransactionRecord transaction, String status);
}

/**
 * Core transaction processor showing SOLID compliance.
 */
public class TransactionProcessor {
    private final LedgerRepository repository;
    private final FeeCalculator feeCalculator;
    private final TransactionNotificationSender notificationSender;

    public TransactionProcessor(
        LedgerRepository repository,
        FeeCalculator feeCalculator,
        TransactionNotificationSender notificationSender
    ) {
        this.repository = Objects.requireNonNull(repository);
        this.feeCalculator = Objects.requireNonNull(feeCalculator);
        this.notificationSender = Objects.requireNonNull(notificationSender);
    }

    /**
     * Processes a transaction. Decoupled from repository, fee, and notification details.
     */
    public void process(TransactionRecord transaction) {
        Objects.requireNonNull(transaction, "Transaction cannot be null");

        // 1. Retrieve accounts from abstraction (DIP)
        LedgerAccount source = repository.findById(transaction.sourceAccountId());
        LedgerAccount destination = repository.findById(transaction.destinationAccountId());

        if (source == null || destination == null) {
            throw new IllegalArgumentException("Source or destination account not found");
        }

        // 2. Calculate fee dynamically (OCP)
        BigDecimal fee = feeCalculator.calculate(transaction);
        BigDecimal totalDebit = transaction.amount().add(fee);

        // 3. Coordinate state transitions on rich domain objects (SRP / LSP)
        // Overdraft check is executed internally within source.debit()
        source.debit(totalDebit);
        destination.credit(transaction.amount());

        // 4. Persist updated states (DIP)
        repository.save(source);
        repository.save(destination);

        // 5. Notify via segregated interface (ISP)
        notificationSender.sendNotification(transaction, "SUCCESS");
    }
}
```
