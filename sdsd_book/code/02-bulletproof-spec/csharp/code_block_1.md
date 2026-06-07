```csharp
// tests/Invariants/TransferInvariants.cs
using Xunit;

public class TransferInvariants : IClassFixture<TestDatabaseFixture>
{
    [Fact]
    public async Task Test_Invariant_Conservation_Of_Mass()
    {
        /*
         * INVARIANT: Sender + Receiver balances must equal the exact same total
         * before and after the transaction, factoring in the network fee.
         */
        var initialA = await _db.GetBalanceAsync(_userA.Id);
        var initialB = await _db.GetBalanceAsync(_userB.Id);
        var initialTotal = initialA + initialB;
        var transferAmount = 100.00m;
        var networkFee = 1.50m;

        // Execute Transfer
        await _transferService.ExecuteFundTransferAsync(_userA.Id, _userB.Id, transferAmount);

        var finalA = await _db.GetBalanceAsync(_userA.Id);
        var finalB = await _db.GetBalanceAsync(_userB.Id);

        // The network fee leaves the system, so we add it back to verify conservation of mass
        var finalTotal = finalA + finalB + networkFee;

        // THE ABSOLUTE TRUTH
        Assert.Equal(initialTotal, finalTotal); // INVARIANT BREACH: Mass not conserved.
        
        // Ensure money actually moved correctly
        Assert.Equal(initialA - transferAmount - networkFee, finalA);
        Assert.Equal(initialB + transferAmount, finalB);
    }
}
```