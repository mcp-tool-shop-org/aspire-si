# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

Only the latest minor version receives security updates.

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report security issues by emailing the maintainers directly or using GitHub's private vulnerability reporting feature:

1. Go to the [Security tab](../../security) of this repository
2. Click "Report a vulnerability"
3. Fill out the form with details about the vulnerability

### What to Include

- Type of vulnerability (e.g., injection, authentication bypass, data exposure)
- Location of the affected code (file path and line numbers if known)
- Step-by-step instructions to reproduce the issue
- Potential impact of the vulnerability
- Any suggested fixes (optional)

### Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Resolution**: Depends on severity and complexity

### Scope

This security policy covers:
- The ASPIRE library code (`aspire/` directory)
- Integration modules (`integrations/` directory)
- Example scripts and configurations
- CI/CD workflows

Out of scope:
- Third-party dependencies (report to upstream maintainers)
- Issues in forked repositories

## Security Best Practices for Users

When using ASPIRE:

1. **API Keys**: Always use environment variables for API keys (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GITHUB_TOKEN`). Never hardcode credentials.

2. **Model Outputs**: Treat model-generated code as untrusted. Always review before execution.

3. **Training Data**: Ensure your training data doesn't contain sensitive information that could be memorized by models.

4. **Dependencies**: Keep dependencies updated. Run `pip install --upgrade aspire-ai` regularly.

## Acknowledgments

We appreciate responsible disclosure and will acknowledge security researchers who report valid vulnerabilities (with permission).
