using System;
using Xunit;
using SdsdSample.Core;

namespace SdsdSample.Tests
{
    public class AuthServiceTests
    {
        [Fact]
        public void TestValidAuthentication()
        {
            var auth = new AuthService();
            Assert.True(auth.Authenticate("tenantA", "admin", "secure123"));
        }

        [Fact]
        public void TestInvalidAuthentication()
        {
            var auth = new AuthService();
            Assert.False(auth.Authenticate("tenantA", "admin", "wrong"));
        }

        [Theory]
        [InlineData(null)]
        [InlineData("")]
        [InlineData("   ")]
        public void TestInvariantTenantIdMissing(string invalidTenantId)
        {
            var auth = new AuthService();
            var ex = Assert.Throws<ArgumentException>(() => auth.Authenticate(invalidTenantId, "admin", "secure123"));
            Assert.Contains("Tenant ID is required", ex.Message);
        }
    }
}
