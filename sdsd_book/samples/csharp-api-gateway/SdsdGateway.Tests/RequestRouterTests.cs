using System;
using Xunit;
using SdsdGateway.Core;

namespace SdsdGateway.Tests
{
    public class RequestRouterTests
    {
        [Fact]
        public void TestValidRouting()
        {
            var router = new RequestRouter();
            Assert.True(router.RouteToLedger("TENANT-A", "TENANT-A", "{ amount: 100 }"));
        }

        [Fact]
        public void TestInvariantIdorPrevention()
        {
            var router = new RequestRouter();
            
            // Malicious payload: User from TENANT-B tries to route to TENANT-A's ledger
            var ex = Assert.Throws<UnauthorizedAccessException>(() => 
                router.RouteToLedger("TENANT-A", "TENANT-B", "{ amount: 100 }")
            );
            
            Assert.Contains("IDOR attempt blocked", ex.Message);
        }
    }
}
