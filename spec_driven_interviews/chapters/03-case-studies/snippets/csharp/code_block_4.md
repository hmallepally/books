```csharp
public class DidConsentRecord 
{
    public string Did { get; }
    private readonly ConcurrentDictionary<string, bool> _consentScopes;

    public DidConsentRecord(string did, Dictionary<string, bool> consentScopes) 
    {
        if (string.IsNullOrEmpty(did) || !did.StartsWith("did:")) 
        {
            throw new ArgumentException("Invalid W3C DID format");
        }
        Did = did;
        _consentScopes = new ConcurrentDictionary<string, bool>(consentScopes);
    }

    public bool HasConsent(string scope) 
    {
        return _consentScopes.TryGetValue(scope, out bool consent) && consent;
    }

    public void RevokeConsent(string scope) 
    {
        _consentScopes[scope] = false;
    }
}
```