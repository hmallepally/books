```java
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import java.math.BigDecimal;

public class TransactionProcessorTest {

    @Test
    public void testSuccessfulTransfer_EnforcesInvariants() {
        // Arrange Mock Dependencies
        LedgerRepository mockRepo = mock(LedgerRepository.class);
        FeeCalculator mockCalculator = mock(FeeCalculator.class);
        TransactionNotificationSender mockSender = mock(TransactionNotificationSender.class);

        LedgerAccount source = new LedgerAccount("acc-source", new BigDecimal("100.00"), "USD");
        LedgerAccount destination = new LedgerAccount("acc-dest", new BigDecimal("50.00"), "USD");

        when(mockRepo.findById("acc-source")).thenReturn(source);
        when(mockRepo.findById("acc-dest")).thenReturn(destination);
        when(mockCalculator.calculateFee(any(BigDecimal.class))).thenReturn(BigDecimal.ZERO);

        TransactionProcessor processor = new TransactionProcessor(mockRepo, mockCalculator, mockSender);

        // Act
        processor.processTransfer("acc-source", "acc-dest", new BigDecimal("30.00"));

        // Assert state invariants updated
        assertEquals(new BigDecimal("70.00"), source.getBalance());
        assertEquals(new BigDecimal("80.00"), destination.getBalance());

        // Assert repository saved both
        verify(mockRepo).save(source);
        verify(mockRepo).save(destination);
        verify(mockSender).sendNotification(any());
    }
}
```