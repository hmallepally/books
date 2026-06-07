```java
// src/test/java/com/aetherfi/transactions/InvariantTransferTest.java
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.transaction.annotation.Transactional;
import java.math.BigDecimal;
import static org.assertj.core.api.Assertions.assertThat;
@SpringBootTest
public class InvariantTransferTest {
    @Autowired
    private TransactionService transactionService;
    @Autowired
    private AccountRepository accountRepository;
    @Test
    @Transactional
    public void testInvariantConservationOfMass() {
        /**
         * INVARIANT: Sender + Receiver balances must equal the exact same total
         * before and after the transaction, factoring in the network fee.
         */
        Account userA = accountRepository.findById(1L).orElseThrow();
        Account userB = accountRepository.findById(2L).orElseThrow();
        BigDecimal initialA = userA.getBalance();
        BigDecimal initialB = userB.getBalance();
        BigDecimal initialTotal = initialA.add(initialB);
        BigDecimal transferAmount = new BigDecimal("100.00");
        BigDecimal networkFee = new BigDecimal("1.50");
        // Execute Transfer
        transactionService.executeFundTransfer(
            userA.getId(), 
            userB.getId(), 
            transferAmount
        );
        Account finalUserA = accountRepository.findById(1L).orElseThrow();
        Account finalUserB = accountRepository.findById(2L).orElseThrow();
        BigDecimal finalA = finalUserA.getBalance();
        BigDecimal finalB = finalUserB.getBalance();
        BigDecimal finalTotal = finalA.add(finalB).add(networkFee);
        // THE ABSOLUTE TRUTH
        assertThat(initialTotal).as("INVARIANT BREACH: Mass not conserved.")
            .isEqualByComparingTo(finalTotal);
        // Ensure money actually moved correctly
        assertThat(finalA).isEqualByComparingTo(initialA.subtract(transferAmount).subtract(networkFee));
        assertThat(finalB).isEqualByComparingTo(initialB.add(transferAmount));
    }
}
```