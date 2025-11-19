# Observability & Monitoring

## Overview

The Enterprise AI Assistant Platform includes comprehensive observability through Opik integration, providing distributed tracing, metrics, and logging for all agent interactions.

## Tracing Architecture

### Instrumentation Points

```
User Request
    │
    ├─► API Endpoint (traced)
    │   └─► Request validation
    │
    ├─► Guardrail Check (traced)
    │   └─► Pattern matching
    │
    ├─► Session Retrieval (traced)
    │   └─► Context building
    │
    ├─► Agent Routing (traced)
    │   └─► Confidence scoring
    │
    ├─► Agent Query (traced)
    │   ├─► HR Agent
    │   ├─► Analytics Agent
    │   │   └─► MCP Tool Calls (traced)
    │   └─► Document Agent
    │       ├─► Document Chunking (traced)
    │       ├─► Embedding Generation (traced)
    │       └─► Similarity Search (traced)
    │
    └─► Response Assembly (traced)
        └─► Session update
```

## Sample Trace

### Trace Metadata

```json
{
  "trace_id": "trace_abc123xyz456",
  "session_id": "sess_789def012",
  "user_id": "user_456",
  "timestamp": "2025-11-19T10:30:45.123Z",
  "duration_ms": 2341,
  "status": "success",
  "agent_used": "hr",
  "model_provider": "bedrock",
  "model_id": "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
}
```

### Trace Timeline

```
┌─────────────────────────────────────────────────────────────┐
│ Trace: HR Agent Query                                       │
│ Duration: 2.34s                                             │
└─────────────────────────────────────────────────────────────┘

0ms     ├─► API Request Received
        │   Input: "Who reports to Jennifer Lee?"
        │
50ms    ├─► Guardrail Check
        │   Status: PASSED
        │   Duration: 50ms
        │
120ms   ├─► Session Retrieval
        │   Session: sess_789def012
        │   History: 2 messages
        │   Duration: 70ms
        │
250ms   ├─► Agent Routing
        │   Decision: hr_agent
        │   Confidence: 0.95
        │   Reasoning: "Organizational structure query"
        │   Duration: 130ms
        │
2100ms  ├─► HR Agent Query
        │   ├─► LLM Call (Bedrock)
        │   │   Model: claude-sonnet-4-5
        │   │   Tokens: 156 input, 89 output
        │   │   Duration: 1850ms
        │   │
        │   └─► Response: "John Smith (Senior Software Engineer)..."
        │
2341ms  └─► Response Sent
            Status: 200 OK
            Total Duration: 2.34s
```

## Metrics Dashboard

### Key Performance Indicators

#### Request Metrics
```
Total Requests (24h):        1,247
Success Rate:                98.4%
Average Latency:             1.8s
P95 Latency:                 3.2s
P99 Latency:                 5.1s
Error Rate:                  1.6%
```

#### Agent Distribution
```
HR Agent:                    42% (524 requests)
Analytics Agent:             31% (387 requests)
Document Agent:              19% (237 requests)
Root (General):              8% (99 requests)
```

#### Token Usage (Cost Tracking)
```
Total Tokens (24h):          1,245,678
Input Tokens:                856,234
Output Tokens:               389,444
Estimated Cost:              $18.45
Average Tokens/Request:      998
```

#### Guardrail Triggers
```
Total Checks:                1,247
Blocked Requests:            23 (1.8%)
Prompt Injection:            15
PII Exposure:                5
Malicious Content:           3
```

## Sample Logs

### Structured Log Format

```json
{
  "timestamp": "2025-11-19T10:30:45.123Z",
  "level": "INFO",
  "service": "enterprise-ai-backend",
  "component": "hr_agent",
  "trace_id": "trace_abc123xyz456",
  "session_id": "sess_789def012",
  "message": "Processing HR query",
  "metadata": {
    "agent": "hr",
    "confidence": 0.95,
    "model": "claude-sonnet-4-5",
    "latency_ms": 1850,
    "tokens_input": 156,
    "tokens_output": 89
  }
}
```

### Log Levels

#### INFO - Normal Operations
```
2025-11-19 10:30:45 INFO [root_chatbot] Processing message for session sess_789def012
2025-11-19 10:30:45 INFO [agent_router] Routing to hr_agent (confidence: 0.95)
2025-11-19 10:30:47 INFO [hr_agent] Successfully received response from hr_agent
```

#### WARNING - Recoverable Issues
```
2025-11-19 10:31:12 WARNING [document_agent] Qdrant connection failed. Using fallback mode.
2025-11-19 10:31:15 WARNING [bedrock_integration] Rate limit hit. Retrying in 2s...
```

#### ERROR - Failures
```
2025-11-19 10:32:01 ERROR [analytics_agent] Exception querying agent: Connection timeout
2025-11-19 10:32:01 ERROR [root_chatbot] All agents failed. Handling with root chatbot.
```

## Opik Dashboard

### Accessing Traces

1. **Cloud Dashboard**: https://www.comet.com/opik
2. **Login**: Use credentials from `.env` file
3. **Workspace**: Your configured workspace name

### Dashboard Views

#### Traces View
- List of all traces with filters
- Search by session_id, user_id, agent
- Sort by duration, timestamp, status

#### Sessions View
- Conversation history per session
- Multi-turn interaction tracking
- User journey analysis

#### Metrics View
- Request volume over time
- Latency percentiles
- Error rate trends
- Token usage and costs

#### Agents View
- Performance by agent type
- Success rates
- Average latency
- Token efficiency

## Alerting

### Alert Conditions

#### Critical Alerts 🔴
```
- Error rate > 5% (5 min window)
- P95 latency > 10s (5 min window)
- Guardrail block rate > 10% (15 min window)
- Service unavailable (health check fails)
```

#### Warning Alerts 🟡
```
- Error rate > 2% (15 min window)
- P95 latency > 5s (15 min window)
- Token usage > $100/day
- Qdrant connection failures
```

### Alert Channels
- Email: ops-team@company.com
- Slack: #ai-platform-alerts
- PagerDuty: Critical alerts only

## Performance Optimization

### Identified Bottlenecks

1. **LLM Latency** (1.8s avg)
   - Optimization: Prompt caching, smaller models for simple queries
   - Target: < 1.5s

2. **Vector Search** (500ms avg)
   - Optimization: Index tuning, reduce k value
   - Target: < 300ms

3. **Session Retrieval** (70ms avg)
   - Optimization: In-memory caching, Redis
   - Target: < 20ms

### Optimization Results

```
Before Optimization:
- Average Latency: 2.8s
- P95 Latency: 5.2s
- Token Usage: 1,200 tokens/request

After Optimization:
- Average Latency: 1.8s (-36%)
- P95 Latency: 3.2s (-38%)
- Token Usage: 998 tokens/request (-17%)
```

## Debugging with Traces

### Example: Investigating Slow Request

**Problem**: User reports slow response for document query

**Step 1**: Find trace by session_id
```
Session: sess_abc123
Trace: trace_xyz789
Duration: 8.2s (slow!)
```

**Step 2**: Analyze trace timeline
```
0ms     API Request
50ms    Guardrail Check (normal)
120ms   Session Retrieval (normal)
250ms   Agent Routing (normal)
8100ms  Document Agent Query (SLOW!)
  ├─ 200ms  Similarity Search (normal)
  ├─ 7800ms LLM Call (SLOW!)
  └─ 100ms  Response Assembly
```

**Step 3**: Identify root cause
- LLM call took 7.8s (expected: 2s)
- Large context window (3,500 tokens)
- Bedrock throttling detected

**Step 4**: Apply fix
- Reduce context window to 2,000 tokens
- Implement sliding window for history
- Add retry logic with backoff

**Result**: Latency reduced to 2.1s ✅

## Cost Tracking

### Token Usage by Agent

```
HR Agent:
- Avg Input Tokens: 180
- Avg Output Tokens: 95
- Cost per Request: $0.012

Analytics Agent:
- Avg Input Tokens: 220
- Avg Output Tokens: 150
- Cost per Request: $0.018

Document Agent:
- Avg Input Tokens: 850 (includes context)
- Avg Output Tokens: 200
- Cost per Request: $0.045
```

### Daily Cost Breakdown

```
Total Daily Cost: $18.45

By Agent:
- HR Agent:        $6.29 (34%)
- Analytics Agent: $6.97 (38%)
- Document Agent:  $5.19 (28%)

By Operation:
- LLM Inference:   $16.20 (88%)
- Embeddings:      $1.85 (10%)
- Vector Storage:  $0.40 (2%)
```

## Best Practices

### Tracing
✅ Trace all agent interactions
✅ Include relevant metadata (model, tokens, confidence)
✅ Use consistent naming conventions
✅ Add tags for filtering (agent:hr, operation:query)

### Logging
✅ Use structured logging (JSON)
✅ Never log PII or credentials
✅ Include trace_id for correlation
✅ Use appropriate log levels

### Metrics
✅ Track latency percentiles (p50, p95, p99)
✅ Monitor error rates by endpoint
✅ Track token usage for cost control
✅ Set up alerts for anomalies

### Cost Optimization
✅ Monitor token usage trends
✅ Optimize prompts for efficiency
✅ Use caching where possible
✅ Consider smaller models for simple queries

---

**Last Updated**: 2025-11-19
**Version**: 1.0.0
