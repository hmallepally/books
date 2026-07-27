```java
package com.aurapay.persistence;

import java.math.BigDecimal;
import java.util.Objects;
import java.util.UUID;

/**
 * Represents a database-mapped Ledger Account Entity with versioning for
 * Optimistic Concurrency Control (OCC).
 */
public class AccountEntity {
    private final UUID id;
    private BigDecimal balance;
    private final String currency;
    private long version; // Enforces OCC state check

    public AccountEntity(UUID id, BigDecimal balance, String currency, long version) {
        this.id = Objects.requireNonNull(id);
        this.balance = Objects.requireNonNull(balance);
        this.currency = Objects.requireNonNull(currency);
        this.version = version;
    }

    public UUID getId() { return id; }
    public BigDecimal getBalance() { return balance; }
    public String getCurrency() { return currency; }
    public long getVersion() { return version; }

    public void updateBalance(BigDecimal newBalance) {
        this.balance = Objects.requireNonNull(newBalance);
    }

    public void incrementVersion() {
        this.version++;
    }
}

/**
 * Repository implementation executing the version check update query.
 */
public class DatabaseLedgerRepository {

    /**
     * Updates the account in the database using a strict version-matching query.
     * Throws an exception if another thread modified the record concurrently.
     */
    public void save(AccountEntity account) {
        // Under the hood, this compiles to the SQL query:
        // UPDATE accounts SET balance = ?, version = version + 1 WHERE id = ? AND version = ?;
        String query = "UPDATE accounts SET balance = :balance, version = :version + 1 " +
                       "WHERE id = :id AND version = :version";

        int rowsUpdated = mockExecuteUpdateQuery(query, account);

        // OCC FAILURE CHECK: If no rows were updated, a concurrent transaction modified the version first.
        if (rowsUpdated == 0) {
            throw new OptimisticLockingFailureException(
                String.format("Optimistic lock conflict on account %s. Outdated version: %d", 
                account.getId(), account.getVersion())
            );
        }

        account.incrementVersion();
    }

    private int mockExecuteUpdateQuery(String query, AccountEntity account) {
        // Simulates the DB executing the update. In a real system, the database engine
        // returns 0 if the WHERE clause (matching ID and version) matches no records.
        return 1; // Returns 1 on success, 0 on concurrent modification conflict
    }
}
```
