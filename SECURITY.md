# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.0.x   | Yes       |
| < 1.0   | No        |

## Reporting a Vulnerability

**Email:** 64996768+mcp-tool-shop@users.noreply.github.com

Please include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact

**Response timeline:**
- Acknowledgment: within 48 hours
- Assessment: within 7 days
- Fix (if confirmed): within 30 days

## Scope

ASPIRE is a **Python ML training framework** for adversarial student-professor learning.
- **Data accessed:** Reads training prompts (JSON/YAML), model checkpoints, and configuration files from local filesystem. Calls external APIs (Anthropic, OpenAI) only when teacher modules are explicitly configured with user-provided API keys.
- **Data NOT accessed:** No telemetry. No user data storage beyond training artifacts. No credential storage — API keys are read from environment variables at runtime.
- **Permissions required:** Read/write access to training data and checkpoint directories. GPU access for model training. Network access only when using API-based teachers (Claude, OpenAI).
