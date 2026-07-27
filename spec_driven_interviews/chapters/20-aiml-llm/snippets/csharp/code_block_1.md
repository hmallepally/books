```csharp
using System;
using System.Text.RegularExpressions;

public class LlmGatewaySecurityFilter 
{
    private static readonly Regex InjectionPattern = new Regex(
        "(ignore all previous instructions|system prompt|bypass validation|reveal key)",
        RegexOptions.IgnoreCase | RegexOptions.Compiled
    );

    public bool ValidatePrompt(string userPrompt) 
    {
        if (string.IsNullOrWhiteSpace(userPrompt)) 
        {
            return false;
        }
        // Fail-fast if malicious injection signature detected
        if (InjectionPattern.IsMatch(userPrompt)) 
        {
            throw new UnauthorizedAccessException("Potential prompt injection attack blocked");
        }
        return true;
    }
}
```