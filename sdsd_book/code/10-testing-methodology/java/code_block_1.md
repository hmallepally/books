```java
import net.jqwik.api.*;
import java.math.BigDecimal;
import static org.assertj.core.api.Assertions.assertThat;

public class ConservationOfMassTest {

    // Invariant: Sender Balance + Receiver Balance must equal initial total (minus network fee)
    @Property
    void testConservationOfMass(
            @ForAll @DoubleRange(min = 0.01, max = 1000000.00) double transferVal,
            @ForAll @DoubleRange(min = 0.00, max = 2000000.00) double initialAVal,
            @ForAll @DoubleRange(min = 0.00, max = 2000000.00) double initialBVal) {
        
        BigDecimal transferAmount = BigDecimal.valueOf(transferVal);
        BigDecimal initialA = BigDecimal.valueOf(initialAVal);
        BigDecimal initialB = BigDecimal.valueOf(initialBVal);

        // Setup state
        Account accountA = new Account(initialA);
        Account accountB = new Account(initialB);
        BigDecimal networkFee = new BigDecimal("1.50");

        // Execute transfer
        try {
            TransferService.processTransfer(accountA, accountB, transferAmount, networkFee);

            // Verify Invariant explicitly
            BigDecimal finalTotal = accountA.getBalance()
                                        .add(accountB.getBalance())
                                        .add(networkFee);
            assertThat(finalTotal).isEqualByComparingTo(initialA.add(initialB));
        } catch (InsufficientFundsException e) {
            // If the transfer fails, balances must remain completely untouched
            assertThat(accountA.getBalance()).isEqualByComparingTo(initialA);
            assertThat(accountB.getBalance()).isEqualByComparingTo(initialB);
        }
    }
}
```