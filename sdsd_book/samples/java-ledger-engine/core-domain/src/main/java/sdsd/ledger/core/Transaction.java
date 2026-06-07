package sdsd.ledger.core;

public class Transaction {
    private final String accountId;
    private final double amount;

    public Transaction(String accountId, double amount) {
        if (accountId == null || accountId.trim().isEmpty()) {
            throw new IllegalArgumentException("Account ID is required.");
        }
        this.accountId = accountId;
        this.amount = amount;
    }

    public String getAccountId() { return accountId; }
    public double getAmount() { return amount; }
}
