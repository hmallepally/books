```java
package com.aurapay.events;

import com.aurapay.domain.TransactionRecord;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Interface defining the Observer contract for transaction events.
 */
public interface TransactionObserver {
    void onTransactionSuccess(TransactionRecord transaction);
    void onTransactionFailed(TransactionRecord transaction, Throwable error);
}

/**
 * Concrete Observer that writes a persistent audit trail for security compliance.
 */
public class AuditTrailObserver implements TransactionObserver {
    
    @Override
    public void onTransactionSuccess(TransactionRecord transaction) {
        System.out.printf("AUDIT SUCCESS: Transaction %s of %s %s from %s to %s registered in immutable log.%n",
            transaction.transactionId(), 
            transaction.amount(), 
            transaction.currency(), 
            transaction.sourceAccountId(), 
            transaction.destinationAccountId());
    }

    @Override
    public void onTransactionFailed(TransactionRecord transaction, Throwable error) {
        System.err.printf("AUDIT FAILURE: Transaction %s failed. Error: %s%n",
            transaction.transactionId(), 
            error.getMessage());
    }
}

/**
 * Subject class managing observers and publishing transaction status updates.
 */
public class TransactionEventPublisher {
    private final List<TransactionObserver> observers = new ArrayList<>();

    public synchronized void registerObserver(TransactionObserver observer) {
        observers.add(Objects.requireNonNull(observer));
    }

    public synchronized void deregisterObserver(TransactionObserver observer) {
        observers.remove(observer);
    }

    public void notifySuccess(TransactionRecord transaction) {
        List<TransactionObserver> targets;
        synchronized (this) {
            targets = new ArrayList<>(observers);
        }
        for (TransactionObserver observer : targets) {
            observer.onTransactionSuccess(transaction);
        }
    }

    public void notifyFailure(TransactionRecord transaction, Throwable error) {
        List<TransactionObserver> targets;
        synchronized (this) {
            targets = new ArrayList<>(observers);
        }
        for (TransactionObserver observer : targets) {
            observer.onTransactionFailed(transaction, error);
        }
    }
}
```
