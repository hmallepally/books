import pytest
from auth_service import AuthService

def test_valid_authentication():
    auth = AuthService()
    assert auth.authenticate("tenantA", "admin", "secure123") is True

def test_invalid_authentication():
    auth = AuthService()
    assert auth.authenticate("tenantA", "admin", "wrong") is False

def test_invariant_tenant_id_missing():
    auth = AuthService()
    with pytest.raises(ValueError, match="Tenant ID is required"):
        auth.authenticate(None, "admin", "secure123")
        
    with pytest.raises(ValueError, match="Tenant ID is required"):
        auth.authenticate("   ", "admin", "secure123")
