# Getting Started with Enterprise AI Assistant Platform

This guide will help you get the backend up and running in minutes.

## 🎯 What You'll Build

A multi-agent AI assistant platform with:
- **HR Agent**: Answers employee and organizational queries
- **Analytics Agent**: Performs calculations and payroll processing
- **Document Agent**: RAG-based document search and Q&A
- **Root Chatbot**: Orchestrates conversations and routes to specialized agents
- **Guardrails**: Content safety and prompt injection prevention
- **Observability**: Full tracing with Opik

## ⚡ Quick Start (5 Minutes)

### Option 1: Automated Setup (Recommended)

```bash
cd backend
./scripts/quick_start.sh
```

This script will:
1. ✅ Check Python version
2. ✅ Create virtual environment
3. ✅ Install dependencies
4. ✅ Create .env from template
5. ✅ Start Qdrant (if Docker is available)

### Option 2: Manual Setup

```bash
# 1. Navigate to backend
cd backend

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your AWS credentials

# 5. Start Qdrant
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant

# 6. Run the application
python main.py
```

## 🔑 Required Configuration

Edit `.env` file with your credentials:

```bash
# Minimum required for AWS Bedrock
USE_BEDROCK=True
AWS_ACCESS_KEY_ID=your_key_here
AWS_SECRET_ACCESS_KEY=your_secret_here
AWS_DEFAULT_REGION=us-east-1
BEDROCK_MODEL_ID=us.anthropic.claude-sonnet-4-5-20250929-v1:0

# Optional: Observability
ENABLE_TRACING=true
OPIK_API_KEY=your_opik_key
```

## ✅ Verify Installation

### 1. Check Health
```bash
curl http://localhost:8000/health
```

Expected output:
```json
{
  "status": "healthy",
  "service": "Enterprise AI Assistant Platform"
}
```

### 2. Test HR Agent
```bash
curl -X POST http://localhost:8000/api/v1/agents/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Who reports to Jennifer Lee?", "session_id": null}'
```

### 3. Run Demo Script
```bash
./scripts/demo_queries.sh
```

This will run through all test scenarios including:
- HR queries (organizational structure, skills)
- Analytics queries (calculations, payroll)
- Document queries (policies, benefits)
- Guardrail tests (prompt injection, content safety)
- Multi-turn conversations

## 📖 Next Steps

### 1. Explore the API
Visit http://localhost:8000/docs for interactive API documentation

### 2. Upload Documents
```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@fixtures/sample_employee_handbook.txt"
```

### 3. Query Documents
```bash
curl -X POST http://localhost:8000/api/v1/agents/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is our PTO policy?", "session_id": null}'
```

### 4. View Traces
If you configured Opik:
1. Visit https://www.comet.com/opik
2. Login with your credentials
3. View traces, metrics, and session history

## 🎓 Learning Path

### Beginner
1. ✅ Complete Quick Start
2. ✅ Test all three agents
3. ✅ Upload a sample document
4. ✅ Review API documentation

### Intermediate
1. 📖 Read [ARCHITECTURE.md](ARCHITECTURE.md) - Understand system design
2. 📖 Read [OBSERVABILITY.md](OBSERVABILITY.md) - Learn about tracing
3. 🔧 Modify agent prompts in `agents/` directory
4. 🔧 Add custom sample data in `fixtures/`

### Advanced
1. 🚀 Implement authentication
2. 🚀 Add rate limiting
3. 🚀 Deploy to production (AWS/GCP/Azure)
4. 🚀 Integrate with your own data sources

## 🐛 Troubleshooting

### Issue: "Connection refused" on port 8000
**Solution**: Check if another process is using port 8000
```bash
lsof -i :8000
# Kill the process or change port in main.py
```

### Issue: "Qdrant connection failed"
**Solution**: Ensure Qdrant is running
```bash
docker ps | grep qdrant
# If not running:
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant
```

### Issue: "AWS credentials not found"
**Solution**: Verify .env file has correct AWS credentials
```bash
cat .env | grep AWS
# Should show your AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY
```

### Issue: "Module not found" errors
**Solution**: Ensure virtual environment is activated and dependencies installed
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: Slow responses (>10s)
**Solution**: Check AWS Bedrock region and model availability
```bash
# Try a different region in .env
AWS_DEFAULT_REGION=us-west-2
```

## 📚 Documentation Index

- **[README.md](README.md)** - Complete documentation
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture and threat model
- **[OBSERVABILITY.md](OBSERVABILITY.md)** - Tracing and monitoring
- **[GETTING_STARTED.md](GETTING_STARTED.md)** - This file

## 💡 Tips & Best Practices

### Development
- Use `DEBUG=True` in .env for detailed logs
- Check logs in console for debugging
- Use Swagger UI at `/docs` for API testing

### Testing
- Run `./scripts/demo_queries.sh` after changes
- Test guardrails with malicious inputs
- Verify fallback behavior (stop Qdrant and test)

### Performance
- Monitor token usage in Opik dashboard
- Optimize prompts to reduce tokens
- Use caching for repeated queries

### Security
- Never commit .env file
- Rotate AWS credentials regularly
- Review guardrail logs for attack attempts

## 🤝 Getting Help

### Resources
- 📖 Full documentation in README.md
- 🔍 API docs at http://localhost:8000/docs
- 🎥 Demo video (see README.md)

### Common Questions

**Q: Can I use Anthropic API instead of AWS Bedrock?**
A: Yes! Set `USE_BEDROCK=False` and provide `api_key` in .env

**Q: How do I add my own documents?**
A: Use the `/api/v1/documents/upload` endpoint with PDF/DOCX/TXT files

**Q: Can I customize agent behavior?**
A: Yes! Edit system prompts in `agents/hr_agent.py`, `analytics_agent.py`, etc.

**Q: How do I deploy to production?**
A: See "Deployment" section in README.md for Docker and AWS examples

**Q: Is there a frontend?**
A: Yes! Check `../frontend/` directory for React/Vite UI

## 🎉 You're Ready!

You now have a fully functional AI assistant platform. Start experimenting with:
- Different types of queries
- Custom documents
- Multi-turn conversations
- Agent routing behavior

Happy building! 🚀

---

**Need help?** Open an issue or contact the development team.
