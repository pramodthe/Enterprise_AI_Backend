#!/bin/bash

# ============================================
# Enterprise AI Assistant Platform
# Quick Start Script
# ============================================

set -e  # Exit on error

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Enterprise AI Assistant Platform - Quick Start            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check Python version
echo -e "${YELLOW}Checking Python version...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is not installed. Please install Python 3.9+${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo -e "${GREEN}✅ Python $PYTHON_VERSION found${NC}"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
else
    echo -e "${GREEN}✅ Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt > /dev/null 2>&1
echo -e "${GREEN}✅ Dependencies installed${NC}"

# Check if .env exists
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}Creating .env file from template...${NC}"
    cp .env.example .env
    echo -e "${GREEN}✅ .env file created${NC}"
    echo -e "${YELLOW}⚠️  Please edit .env file with your credentials before running!${NC}"
    echo ""
    echo "Required configuration:"
    echo "  - AWS_ACCESS_KEY_ID"
    echo "  - AWS_SECRET_ACCESS_KEY"
    echo "  - OPIK_API_KEY (optional, for observability)"
    echo "  - OPIK_WORKSPACE (optional, for observability)"
    echo ""
else
    echo -e "${GREEN}✅ .env file already exists${NC}"
fi

# Check if Docker is running
echo -e "${YELLOW}Checking Docker...${NC}"
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed${NC}"
    echo "Please install Docker to run Qdrant vector database"
    echo "Visit: https://docs.docker.com/get-docker/"
else
    echo -e "${GREEN}✅ Docker found${NC}"
    
    # Check if Qdrant is running
    if docker ps | grep -q qdrant; then
        echo -e "${GREEN}✅ Qdrant is already running${NC}"
    else
        echo -e "${YELLOW}Starting Qdrant vector database...${NC}"
        docker run -d -p 6333:6333 --name qdrant qdrant/qdrant > /dev/null 2>&1
        echo -e "${GREEN}✅ Qdrant started on port 6333${NC}"
    fi
fi

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Setup Complete! 🎉                                        ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Next steps:"
echo ""
echo "1. Configure your .env file with credentials:"
echo -e "   ${BLUE}nano .env${NC}"
echo ""
echo "2. Start the backend server:"
echo -e "   ${BLUE}python main.py${NC}"
echo ""
echo "3. Test the API:"
echo -e "   ${BLUE}curl http://localhost:8000/health${NC}"
echo ""
echo "4. Run demo queries:"
echo -e "   ${BLUE}./scripts/demo_queries.sh${NC}"
echo ""
echo "5. View API documentation:"
echo -e "   ${BLUE}http://localhost:8000/docs${NC}"
echo ""
echo -e "${YELLOW}Note: Make sure to configure AWS credentials in .env before starting!${NC}"
echo ""
