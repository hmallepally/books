package sdsd.ledger.service;

import sdsd.ledger.core.Account;
import sdsd.ledger.core.Transaction;
import java.util.HashSet;
import java.util.Set;

public class LedgerService {
    private final Set<String> processedTransactionIds = new HashSet<>();

    public void processTransaction(Account account, Transaction tx, String transactionId) {
        // SDSD Invariant 1: No double spending
        if (processedTransactionIds.contains(transactionId)) {
            throw new IllegalStateException("Duplicate transaction detected: " + transactionId);
        }

        // SDSD Invariant 2: Tenant Isolation Check
        if (!account.getId().equals(tx.getAccountId())) {
            throw new SecurityException("Transaction account ID does not match the target account.");
        }

        // SDSD Invariant 3: No negative balances
        double newBalance = account.getBalance() + tx.getAmount();
        if (newBalance < 0) {
            throw new IllegalArgumentException("Insufficient funds. Balance cannot drop below zero.");
        }

        account.apply(tx);
        processedTransactionIds.add(transactionId);
    }
}
