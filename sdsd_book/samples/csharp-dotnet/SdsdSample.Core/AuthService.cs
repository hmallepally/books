using System;

namespace SdsdSample.Core
{
    public class AuthService
    {
        public bool Authenticate(string tenantId, string username, string password)
        {
            // SDSD Invariant: Tenant Isolation
            if (string.IsNullOrWhiteSpace(tenantId))
            {
                throw new ArgumentException("Tenant ID is required for isolation.", nameof(tenantId));
            }
            
            // Dummy logic
            if (username == "admin" && password == "secure123")
            {
                return true;
            }
            return false;
        }
    }
}
