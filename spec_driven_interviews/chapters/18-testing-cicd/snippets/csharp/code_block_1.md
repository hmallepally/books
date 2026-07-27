```csharp
using Moq;
using Xunit;

public class TransactionProcessorTests
{
    [Fact]
    public void TestSuccessfulTransfer_EnforcesInvariants()
    {
        // Arrange Mock Dependencies
        var mockRepo = new Mock<ILedgerRepository>();
        var mockCalculator = new Mock<IFeeCalculator>();
        var mockSender = new Mock<ITransactionNotificationSender>();

        var source = new LedgerAccount("acc-source", 100.00m, "USD");
        var destination = new LedgerAccount("acc-dest", 50.00m, "USD");

        mockRepo.Setup(r => r.FindById("acc-source")).Returns(source);
        mockRepo.Setup(r => r.FindById("acc-dest")).Returns(destination);
        mockCalculator.Setup(c => c.CalculateFee(It.IsAny<decimal>())).Returns(0.00m);

        var processor = new TransactionProcessor(mockRepo.Object, mockCalculator.Object, mockSender.Object);

        // Act
        processor.ProcessTransfer("acc-source", "acc-dest", 30.00m);

        // Assert state invariants updated
        Assert.Equal(70.00m, source.Balance);
        Assert.Equal(80.00m, destination.Balance);

        // Assert repository saved both
        mockRepo.Verify(r => r.Save(source), Times.Once);
        mockRepo.Verify(r => r.Save(destination), Times.Once);
        mockSender.Verify(s => s.SendNotification(It.IsAny<TransactionEvent>()), Times.Once);
    }
}
```