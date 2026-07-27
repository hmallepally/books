```python
class DidConsentRecord:
    def __init__(self, did: str, consent_scopes: dict[str, bool]):
        if not did or not did.startswith("did:"):
            raise ValueError("Invalid W3C DID format")
        self.did = did
        self._consent_scopes = dict(consent_scopes)

    def has_consent(self, scope: str) -> bool:
        return self._consent_scopes.get(scope, False)

    def revoke_consent(self, scope: str) -> None:
        self._consent_scopes[scope] = False
```