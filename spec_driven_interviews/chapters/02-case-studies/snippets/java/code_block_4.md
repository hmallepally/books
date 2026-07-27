```java
public class DidConsentRecord {
    private final String did;
    private final Map<String, Boolean> consentScopes;

    public DidConsentRecord(String did, Map<String, Boolean> consentScopes) {
        if (did == null || !did.startsWith("did:")) {
            throw new IllegalArgumentException("Invalid W3C DID format");
        }
        this.did = did;
        this.consentScopes = new ConcurrentHashMap<>(consentScopes);
    }

    public String getDid() { return did; }
    
    public boolean hasConsent(String scope) {
        return consentScopes.getOrDefault(scope, false);
    }

    public void revokeConsent(String scope) {
        consentScopes.put(scope, false);
    }
}
```