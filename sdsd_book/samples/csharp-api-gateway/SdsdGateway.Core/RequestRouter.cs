using System;

namespace SdsdGateway.Core
{
    public class RequestRouter
    {
        public bool RouteToLedger(string requestTenantId, string userTenantContext, string payload)
        {
            // SDSD Invariant: Strict Tenant Isolation (IDOR Prevention)
            if (string.IsNullOrWhiteSpace(requestTenantId) || string.IsNullOrWhiteSpace(userTenantContext))
            {
                throw new ArgumentException("Tenant contexts cannot be empty.");
            }

            if (requestTenantId != userTenantContext)
            {
                throw new UnauthorizedAccessException($"IDOR attempt blocked. User from {userTenantContext} tried to access {requestTenantId}.");
            }

            // Simulate successful routing
            return true;
        }
    }
}
