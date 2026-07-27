```java
package com.aurapay.domain;

import java.math.BigDecimal;
import java.util.Objects;
import java.util.UUID;

/**
 * Represents a stateful Ledger Account in AuraPay, enforcing business invariants
 * during state transitions.
 */
public class LedgerAccount {
    private final UUID accountId;
    private final String currency;
    private BigDecimal balance;
    private final BigDecimal overdraftLimit;

    public LedgerAccount(UUID accountId, String currency, BigDecimal initialBalance, BigDecimal overdraftLimit) {
        this.accountId = Objects.requireNonNull(accountId, "Account ID cannot be null");
        this.currency = Objects.requireNonNull(currency, "Currency cannot be null");
        Objects.requireNonNull(initialBalance, "Initial balance cannot be null");
        Objects.requireNonNull(overdraftLimit, "Overdraft limit cannot be negative");

        if (overdraftLimit.compareTo(BigDecimal.ZERO) < 0) {
            throw new IllegalArgumentException("Overdraft limit cannot be negative");
        }
        if (initialBalance.add(overdraftLimit).compareTo(BigDecimal.ZERO) < 0) {
            throw new IllegalArgumentException("Initial balance violates the overdraft limit");
        }

        this.balance = initialBalance;
        this.overdraftLimit = overdraftLimit;
    }

    public UUID getAccountId() { return accountId; }
    public String getCurrency() { return currency; }
    public synchronized BigDecimal getBalance() { return balance; }

    /**
     * Credits the account (adds funds). Enforces positive credit amount.
     */
    public synchronized void credit(BigDecimal amount) {
        Objects.requireNonNull(amount, "Credit amount cannot be null");
        if (amount.compareTo(BigDecimal.ZERO) <= 0) {
            throw new IllegalArgumentException("Credit amount must be positive");
        }
        this.balance = this.balance.add(amount);
    }

    /**
     * Debits the account (removes funds). Enforces balance invariants and overdraft limits.
     */
    public synchronized void debit(BigDecimal amount) {
        Objects.requireNonNull(amount, "Debit amount cannot be null");
        if (amount.compareTo(BigDecimal.ZERO) <= 0) {
            throw new IllegalArgumentException("Debit amount must be positive");
        }
        
        BigDecimal newBalance = this.balance.subtract(amount);
        // INVARIANT ENFORCEMENT: Ensure the account does not exceed its overdraft limit
        if (newBalance.add(this.overdraftLimit).compareTo(BigDecimal.ZERO) < 0) {
            throw new InsufficientFundsException(
                String.format("Debit of %s exceeds account overdraft boundary. Balance: %s, Limit: -%s", 
                amount, balance, overdraftLimit)
            );
        }
        this.balance = newBalance;
    }
}
```
