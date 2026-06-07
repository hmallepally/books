class AuthService:
    def authenticate(self, tenant_id: str, username: str, password: str) -> bool:
        # SDSD Invariant: Tenant Isolation
        if not tenant_id or not tenant_id.strip():
            raise ValueError("Tenant ID is required for isolation.")
            
        # Dummy logic
        if username == "admin" and password == "secure123":
            return True
        return False
