```java
package com.aurapay.domain;

import java.math.BigDecimal;
import java.util.Objects;

/**
 * Demonstrates a rich domain model encapsulating transfer logic and enforcing 
 * cross-entity invariants.
 */
public class LedgerAccount {
    private final String accountId;
    private final String currency;
    private BigDecimal balance;
    private final BigDecimal overdraftLimit;

    public LedgerAccount(String accountId, String currency, BigDecimal initialBalance, BigDecimal overdraftLimit) {
        this.accountId = Objects.requireNonNull(accountId);
        this.currency = Objects.requireNonNull(currency);
        this.balance = Objects.requireNonNull(initialBalance);
        this.overdraftLimit = Objects.requireNonNull(overdraftLimit);
    }

    public synchronized BigDecimal getBalance() { return balance; }
    public String getCurrency() { return currency; }

    public synchronized void debit(BigDecimal amount) {
        if (amount.compareTo(BigDecimal.ZERO) <= 0) {
            throw new IllegalArgumentException("Debit amount must be positive");
        }
        BigDecimal newBalance = this.balance.subtract(amount);
        if (newBalance.add(this.overdraftLimit).compareTo(BigDecimal.ZERO) < 0) {
            throw new InsufficientFundsException("Overdraft limit exceeded");
        }
        this.balance = newBalance;
    }

    public synchronized void credit(BigDecimal amount) {
        if (amount.compareTo(BigDecimal.ZERO) <= 0) {
            throw new IllegalArgumentException("Credit amount must be positive");
        }
        this.balance = this.balance.add(amount);
    }

    /**
     * Executes a thread-safe transfer to a target account, enforcing business invariants.
     * Prevents mismatched currencies (pre-condition) and double-debiting.
     */
    public void transferTo(LedgerAccount target, BigDecimal amount) {
        Objects.requireNonNull(target, "Destination account cannot be null");
        Objects.requireNonNull(amount, "Transfer amount cannot be null");

        // PRE-CONDITION ENFORCEMENT: Currency matching
        if (!this.currency.equals(target.getCurrency())) {
            throw new CurrencyMismatchException(
                String.format("Cannot transfer between mismatched currencies: %s and %s", 
                this.currency, target.getCurrency())
            );
        }

        // PRE-CONDITION ENFORCEMENT: Self-transfer check
        if (this.accountId.equals(target.accountId)) {
            throw new IllegalArgumentException("Cannot transfer to the same account");
        }

        // To prevent deadlocks, lock accounts in a stable global order
        LedgerAccount firstLock = this.accountId.compareTo(target.accountId) < 0 ? this : target;
        LedgerAccount secondLock = firstLock == this ? target : this;

        synchronized (firstLock) {
            synchronized (secondLock) {
                // Execute atomic debit-credit sequence
                this.debit(amount);
                target.credit(amount);
            }
        }
    }
}
```
