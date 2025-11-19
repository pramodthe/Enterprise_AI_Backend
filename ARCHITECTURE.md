# Enterprise AI Assistant Platform - Architecture & Threat Model

## System Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Client Layer (Frontend)                       │
│                     React + Vite + Tailwind                      │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTPS/REST API
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                      API Gateway Layer                           │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  FastAPI Application (main.py)                           │   │
│  │  - CORS Middleware                                       │   │
│  │  - Request Validation                                    │   │
│  │  - Error Handling                                        │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                   Orchestration Layer                            │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Root Chatbot (root_chatbot.py)                          │   │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────┐ │   │
│  │  │ Session        │  │ Agent Router   │  │ Guardrails │ │   │
│  │  │ Manager        │  │                │  │            │ │   │
│  │  └────────────────┘  └────────────────┘  └────────────┘ │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
┌───────▼────────┐  ┌────────▼────────┐  ┌───────▼──────────┐
│   HR Agent     │  │ Analytics Agent │  │ Document Agent   │
│                │  │                 │  │                  │
│ - Employee DB  │  │ - MCP Tools     │  │ - RAG System     │
│ - Org Chart    │  │ - Calculations  │  │ - Vector Search  │
│ - Skills       │  │ - Payroll       │  │ - Embeddings     │
└───────┬────────┘  └────────┬────────┘  └───────┬──────────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
┌───────▼────────┐  ┌────────▼────────┐  ┌───────▼──────────┐
│  AWS Bedrock   │  │  Qdrant Vector  │  │     Opik         │
│  Claude LLM    │  │  Database       │  │   Observability  │
└────────────────┘  └─────────────────┘  └──────────────────┘
```

## Component Details

### 1. API Gateway Layer
**Responsibility**: Request handling, validation, routing
**Technology**: FastAPI, Pydantic
**Endpoints**:
- `POST /api/v1/agents/chat` - Main chat interface
- `POST /api/v1/documents/upload` - Document ingestion
- `POST /api/v1/documents/query` - Document search
- `GET /health` - Health check

### 2. Orchestration Layer
**Responsibility**: Conversation management, agent coordination, safety
**Components**:
- **Root Chatbot**: Main orchestrator
- **Session Manager**: Conversation state and history
- **Agent Router**: Intelligent query routing with confidence scoring
- **Guardrails**: Content safety and prompt injection prevention

### 3. Agent Layer
**Responsibility**: Specialized domain expertise
**Agents**:
- **HR Agent**: Employee data, organizational structure, skills
- **Analytics Agent**: Calculations, metrics, payroll processing
- **Document Agent**: RAG-based document search and Q&A

### 4. External Services
- **AWS Bedrock**: LLM inference (Claude models)
- **Qdrant**: Vector database for document embeddings
- **Opik**: Distributed tracing and observability

## Data Flow

### Typical Request Flow

```
1. User Query
   ↓
2. API Gateway (validation)
   ↓
3. Guardrails (safety check)
   ↓
4. Session Manager (context retrieval)
   ↓
5. Agent Router (routing decision)
   ↓
6. Specialized Agent (processing)
   ↓
7. LLM/Tools (generation)
   ↓
8. Response Assembly
   ↓
9. Session Update
   ↓
10. Client Response
```

### Document Upload Flow

```
1. File Upload (PDF/DOCX/TXT)
   ↓
2. Document Parsing
   ↓
3. Text Chunking (1000 chars, 200 overlap)
   ↓
4. Embedding Generation (Bedrock/OpenAI)
   ↓
5. Vector Storage (Qdrant)
   ↓
6. Confirmation Response
```

### RAG Query Flow

```
1. User Query
   ↓
2. Query Embedding
   ↓
3. Similarity Search (Qdrant, k=3)
   ↓
4. Context Assembly
   ↓
5. LLM Generation (with context)
   ↓
6. Response with Sources
```

## Trust Boundaries

### Boundary 1: Internet → API Gateway
**Threat**: Malicious requests, DDoS, injection attacks
**Controls**:
- CORS policy (restrict origins)
- Input validation (Pydantic models)
- Rate limiting (to be implemented)
- HTTPS in production

### Boundary 2: API Gateway → Guardrails
**Threat**: Prompt injection, jailbreaking, PII exposure
**Controls**:
- Pattern-based detection
- Content filtering
- Request sanitization
- Logging (without PII)

### Boundary 3: Guardrails → Agents
**Threat**: Agent manipulation, resource exhaustion
**Controls**:
- Agent isolation
- Timeout enforcement
- Resource limits
- Fallback mechanisms

### Boundary 4: Agents → LLM Services
**Threat**: Credential theft, unauthorized access, data leakage
**Controls**:
- API key rotation
- Request signing (AWS SigV4)
- Encrypted transmission
- No credentials in logs

### Boundary 5: Agents → Vector Database
**Threat**: Data poisoning, unauthorized retrieval
**Controls**:
- Network isolation
- Access control (to be implemented)
- Document validation
- Audit logging

## Threat Model

### STRIDE Analysis

#### Spoofing
- **Threat**: Impersonation of legitimate users
- **Mitigation**: Session tokens, authentication (to be added)
- **Priority**: High

#### Tampering
- **Threat**: Modification of requests/responses in transit
- **Mitigation**: HTTPS, request signing
- **Priority**: High

#### Repudiation
- **Threat**: Users deny actions
- **Mitigation**: Audit logging, tracing (Opik)
- **Priority**: Medium

#### Information Disclosure
- **Threat**: Exposure of sensitive data (PII, credentials)
- **Mitigation**: Guardrails, no secrets in code, encrypted storage
- **Priority**: High

#### Denial of Service
- **Threat**: Resource exhaustion, service unavailability
- **Mitigation**: Rate limiting, timeouts, graceful degradation
- **Priority**: Medium

#### Elevation of Privilege
- **Threat**: Unauthorized access to admin functions
- **Mitigation**: RBAC (to be implemented), least privilege
- **Priority**: High

## Security Controls

### Implemented ✅
1. Environment-based secrets management
2. Input validation on all endpoints
3. Guardrails for content safety
4. Structured logging (no PII)
5. Error handling with safe messages
6. Graceful degradation for service failures
7. Retry logic with exponential backoff

### Planned 🔄
1. Authentication & authorization (JWT)
2. Rate limiting per user/IP
3. HTTPS/TLS in production
4. Database encryption at rest
5. API key rotation automation
6. Advanced threat detection

### Recommended 💡
1. Web Application Firewall (WAF)
2. DDoS protection (CloudFlare/AWS Shield)
3. Security scanning (SAST/DAST)
4. Penetration testing
5. Incident response plan

## Performance Considerations

### Latency Targets
- API response: < 2s (p95)
- Document upload: < 5s for 10MB file
- Vector search: < 500ms
- LLM generation: < 3s

### Scalability
- Horizontal scaling: FastAPI workers
- Caching: Session data, embeddings
- Async processing: Document ingestion
- Load balancing: Multiple instances

### Monitoring
- Request latency (p50, p95, p99)
- Error rates by endpoint
- Token usage and costs
- Agent routing distribution
- Guardrail trigger frequency

## Disaster Recovery

### Backup Strategy
- Vector database: Daily snapshots
- Session data: Persistent storage (Supabase)
- Configuration: Version control (Git)
- Logs: Centralized logging (CloudWatch)

### Failure Scenarios
1. **LLM Service Down**: Fallback to cached responses
2. **Vector DB Down**: Use sample data, notify user
3. **Agent Failure**: Route to fallback agent or root
4. **Network Issues**: Retry with exponential backoff

## Compliance Considerations

### Data Privacy
- GDPR: User data deletion, consent management
- CCPA: Data access requests, opt-out
- HIPAA: PHI handling (if applicable)

### Audit Requirements
- Request/response logging
- User action tracking
- Access control logs
- Security event monitoring

---

**Last Updated**: 2025-11-19
**Version**: 1.0.0
