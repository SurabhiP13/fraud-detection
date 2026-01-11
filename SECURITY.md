# Security Configuration Guide

## ⚠️ Important Security Notes

This application includes default credentials and configurations for **development purposes only**. 
**DO NOT use these in production without changing them first!**

## Production Security Checklist

### 1. Environment Variables & Secrets

- [ ] Copy `.env.example` to `.env` and update all credentials
- [ ] Generate strong, unique passwords for all services
- [ ] Never commit `.env` file to version control
- [ ] Use secret management solutions (AWS Secrets Manager, HashiCorp Vault, etc.)

### 2. Database Security

**Local/Docker:**
```bash
# Change default PostgreSQL credentials in .env
POSTGRES_USER=your_unique_user
POSTGRES_PASSWORD=generate_strong_password_here
POSTGRES_DB=fraud_detection_prod
```

**Kubernetes:**
```bash
# Generate base64 encoded secrets
echo -n 'your_username' | base64
echo -n 'your_strong_password' | base64

# Update k8s/postgres.yaml with new values
```

### 3. API Security

- [ ] Enable HTTPS/TLS for all API endpoints
- [ ] Implement authentication (OAuth2, JWT, API keys)
- [ ] Add rate limiting to prevent abuse
- [ ] Configure CORS for specific origins only
- [ ] Enable API key rotation

**Update backend/main.py:**
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Specific domains only
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Limit methods
    allow_headers=["*"],
)
```

### 4. Airflow Security

- [ ] Change default admin credentials
- [ ] Enable authentication backend
- [ ] Use Fernet key for encryption
- [ ] Secure connections to external systems

**Airflow Configuration:**
```bash
# Generate Fernet key
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

# Set in airflow.cfg or environment
AIRFLOW__CORE__FERNET_KEY=your_generated_key
```

### 5. Docker Security

- [ ] Don't run containers as root
- [ ] Use secrets management (Docker secrets, swarm secrets)
- [ ] Scan images for vulnerabilities
- [ ] Use minimal base images
- [ ] Keep images up to date

**Docker Compose with Secrets:**
```yaml
services:
  postgres:
    environment:
      POSTGRES_USER_FILE: /run/secrets/postgres_user
      POSTGRES_PASSWORD_FILE: /run/secrets/postgres_password
    secrets:
      - postgres_user
      - postgres_password

secrets:
  postgres_user:
    file: ./secrets/postgres_user.txt
  postgres_password:
    file: ./secrets/postgres_password.txt
```

### 6. Kubernetes Security

- [ ] Use Kubernetes Secrets (not ConfigMaps for sensitive data)
- [ ] Enable RBAC (Role-Based Access Control)
- [ ] Use Network Policies to restrict traffic
- [ ] Enable Pod Security Standards
- [ ] Use external secret management (External Secrets Operator)

**Example with External Secrets:**
```yaml
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: vault-backend
spec:
  provider:
    vault:
      server: "https://vault.example.com"
      path: "secret"
```

### 7. ML Model Security

- [ ] Validate all input data
- [ ] Implement model versioning
- [ ] Monitor for model poisoning attempts
- [ ] Secure model storage
- [ ] Log all predictions for audit

### 8. Network Security

- [ ] Use private networks for inter-service communication
- [ ] Enable TLS for all connections
- [ ] Implement firewall rules
- [ ] Use VPN for remote access
- [ ] Monitor network traffic

### 9. Logging & Monitoring

- [ ] Enable audit logging for all transactions
- [ ] Monitor for suspicious activities
- [ ] Set up alerts for security events
- [ ] Implement log rotation and retention
- [ ] Use centralized logging (ELK, Splunk, etc.)

### 10. Data Protection

- [ ] Encrypt data at rest
- [ ] Encrypt data in transit (TLS)
- [ ] Implement data anonymization
- [ ] Follow GDPR/CCPA requirements
- [ ] Regular security audits

## Environment Variable Template

Create a `.env` file with these variables (change all values!):

```bash
# Database - CHANGE THESE!
POSTGRES_USER=your_unique_username
POSTGRES_PASSWORD=generate_strong_password_32chars_minimum
POSTGRES_DB=fraud_detection_prod
DATABASE_URL=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}

# Airflow - CHANGE THESE!
AIRFLOW__CORE__FERNET_KEY=generate_fernet_key_here
AIRFLOW__WEBSERVER__SECRET_KEY=generate_secret_key_here
AIRFLOW_ADMIN_USERNAME=your_admin_user
AIRFLOW_ADMIN_PASSWORD=your_admin_password

# API Security
API_SECRET_KEY=generate_jwt_secret_here
ALLOWED_ORIGINS=https://yourdomain.com,https://app.yourdomain.com

# Optional: External Services
SMTP_HOST=smtp.example.com
SMTP_USER=notifications@example.com
SMTP_PASSWORD=your_smtp_password
```

## Password Generation

Generate strong passwords using:

```bash
# Linux/Mac
openssl rand -base64 32

# Python
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Or use a password manager
```

## Security Scanning

### Dependency Scanning
```bash
# Install safety
pip install safety

# Scan dependencies
safety check -r requirements.txt
```

### Container Scanning
```bash
# Using Trivy
trivy image fraud-detection-backend:latest

# Using Docker scan
docker scan fraud-detection-backend:latest
```

### Code Scanning
```bash
# Using bandit for Python
pip install bandit
bandit -r backend/
```

## Regular Security Maintenance

1. **Weekly:**
   - Review access logs
   - Check for failed login attempts
   - Update dependencies if needed

2. **Monthly:**
   - Rotate passwords and API keys
   - Review user access permissions
   - Update security patches

3. **Quarterly:**
   - Full security audit
   - Penetration testing
   - Review and update security policies

## Incident Response

In case of a security incident:

1. Isolate affected systems
2. Preserve evidence (logs, etc.)
3. Notify stakeholders
4. Change all credentials
5. Investigate root cause
6. Implement fixes
7. Document lessons learned

## Compliance

Ensure compliance with:
- **PCI DSS** (if handling payment data)
- **GDPR** (if processing EU data)
- **CCPA** (if processing California data)
- **SOC 2** (for service organizations)
- **ISO 27001** (information security management)

## Support

For security concerns or to report vulnerabilities:
- Email: security@yourdomain.com
- Create a private security advisory on GitHub
- Use responsible disclosure practices

---

**Remember: Security is an ongoing process, not a one-time setup!**
