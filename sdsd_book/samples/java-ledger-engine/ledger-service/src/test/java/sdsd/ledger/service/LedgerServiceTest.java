package sdsd.ledger.service;

import sdsd.ledger.core.Account;
import sdsd.ledger.core.Transaction;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class LedgerServiceTest {

    @Test
    void testValidDeposit() {
        Account acc = new Account("ACC-1", 100.0);
        Transaction tx = new Transaction("ACC-1", 50.0);
        LedgerService service = new LedgerService();
        
        service.processTransaction(acc, tx, "TX-1");
        assertEquals(150.0, acc.getBalance());
    }

    @Test
    void testNegativeBalanceInvariant() {
        Account acc = new Account("ACC-1", 100.0);
        Transaction tx = new Transaction("ACC-1", -150.0);
        LedgerService service = new LedgerService();
        
        assertThrows(IllegalArgumentException.class, () -> {
            service.processTransaction(acc, tx, "TX-1");
        });
        assertEquals(100.0, acc.getBalance(), "Balance should remain unchanged");
    }

    @Test
    void testDoubleSpendInvariant() {
        Account acc = new Account("ACC-1", 100.0);
        Transaction tx1 = new Transaction("ACC-1", -50.0);
        Transaction tx2 = new Transaction("ACC-1", -50.0);
        LedgerService service = new LedgerService();
        
        service.processTransaction(acc, tx1, "TX-1");
        
        assertThrows(IllegalStateException.class, () -> {
            service.processTransaction(acc, tx2, "TX-1"); // Reusing TX-1
        });
        assertEquals(50.0, acc.getBalance(), "Second transaction should be blocked");
    }

    @Test
    void testAccountMismatchInvariant() {
        Account acc = new Account("ACC-1", 100.0);
        Transaction tx = new Transaction("ACC-2", 50.0); // Malicious payload for ACC-2
        LedgerService service = new LedgerService();
        
        assertThrows(SecurityException.class, () -> {
            service.processTransaction(acc, tx, "TX-1");
        });
        assertEquals(100.0, acc.getBalance());
    }

    @Test
    void testExactZeroBalanceInvariant() {
        // This test kills the ConditionalsBoundaryMutator that changes < 0 to <= 0
        Account acc = new Account("ACC-1", 100.0);
        Transaction tx = new Transaction("ACC-1", -100.0);
        LedgerService service = new LedgerService();
        
        service.processTransaction(acc, tx, "TX-1");
        assertEquals(0.0, acc.getBalance(), "Balance can drop to exactly zero");
    }
}
