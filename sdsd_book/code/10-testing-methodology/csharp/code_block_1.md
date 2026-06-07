```csharp
// tests/Transfer/ConservationTest.cs
using FsCheck;
using FsCheck.Xunit;
using System;

public class ConservationTest
{
    // Invariant: Sender Balance + Receiver Balance must equal initial total (minus network fee)
    [Property(MaxTest = 1000)]
    public Property Test_Conservation_Of_Mass(decimal transferAmount, decimal initialA, decimal initialB)
    {
        // Setup state
        var accountA = new Account { Balance = Math.Abs(initialA) };
        var accountB = new Account { Balance = Math.Abs(initialB) };
        var networkFee = 1.50m;
        var transferAmt = Math.Abs(transferAmount);

        // Execute transfer
        try
        {
            TransferService.ProcessTransfer(accountA, accountB, transferAmt, networkFee);

            // Verify Invariant explicitly
            var finalTotal = accountA.Balance + accountB.Balance + networkFee;
            return (finalTotal == (Math.Abs(initialA) + Math.Abs(initialB))).ToProperty();
        }
        catch (InsufficientFundsException)
        {
            // If the transfer fails, balances must remain completely untouched
            var pristine = accountA.Balance == Math.Abs(initialA) && accountB.Balance == Math.Abs(initialB);
            return pristine.ToProperty();
        }
    }
}
```