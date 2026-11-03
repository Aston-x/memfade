# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.4.x   | :white_check_mark: |
| < 0.4   | :x:                |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability, please report it responsibly.

### How to Report

**Please do NOT open a public GitHub issue for security vulnerabilities.**

Instead, please email the maintainers directly or use GitHub's private vulnerability reporting feature:

1. Go to the repository's Security tab
2. Click "Report a vulnerability"
3. Provide details about the vulnerability

### What to Include

- Description of the vulnerability
- Steps to reproduce (if applicable)
- Potential impact
- Suggested fix (if you have one)

### What to Expect

- **Acknowledgment**: We'll acknowledge receipt within 48 hours
- **Assessment**: We'll assess the vulnerability within 7 days
- **Updates**: We'll keep you informed of our progress
- **Resolution**: We'll work to release a fix as soon as possible

### Disclosure Policy

We follow coordinated disclosure:

1. You report the vulnerability privately
2. We work on a fix
3. We release the fix
4. We publicly disclose the vulnerability after users have had time to update

## Security Considerations

### Memory Isolation

memfade enforces strict session isolation by default. Cross-scope sharing requires explicit opt-in via `cross_scope=True`. This is a security feature to prevent private conversation data from leaking into group contexts.

### Data Storage

- Memories are stored in a local SQLite database
- No data is sent to external services by default
- Vector embeddings are computed locally if enabled
- LLM calls (for extraction/summarization) go to the configured provider

### Adapter Security

Each adapter runs in the host bot's process and inherits its security model. Adapters do not:
- Execute arbitrary code
- Access the network (except for configured LLM/embedding endpoints)
- Modify files outside the configured database path

## Known Limitations

- **No authentication**: memfade is designed for single-user or trusted environments
- **No encryption at rest**: The SQLite database is not encrypted
- **No access control**: Any process with database access can read/write memories

These are intentional design choices for a local-first plugin. If you need multi-tenant security, consider mem0 or Hindsight.

## Security Best Practices

When deploying memfade:

1. **Restrict database access**: Set appropriate file permissions on the SQLite database
2. **Use trusted LLM providers**: Memory extraction sends conversation data to the configured LLM
3. **Review memory content**: Periodically audit stored memories for sensitive information
4. **Keep dependencies updated**: While the core has no dependencies, adapters may
5. **Monitor logs**: Watch for unexpected memory writes or access patterns
