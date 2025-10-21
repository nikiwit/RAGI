#!/bin/bash
# RAGI Setup Script - Phase 1 & 2 Automation
# This script automates the directory setup and file copying from SARA to RAGI

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Paths
SARA_PATH="/Users/kita/SARA"
RAGI_PATH="$HOME/Desktop/RAGI"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}RAGI Setup Script - Phase 1 & 2${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if we're in the right directory
if [ "$PWD" != "$RAGI_PATH" ]; then
    echo -e "${RED}Error: Please run this script from $RAGI_PATH${NC}"
    echo "Run: cd $RAGI_PATH && bash setup_ragi.sh"
    exit 1
fi

# Check if SARA exists
if [ ! -d "$SARA_PATH" ]; then
    echo -e "${RED}Error: SARA directory not found at $SARA_PATH${NC}"
    exit 1
fi

echo -e "${GREEN}✓ SARA directory found${NC}"
echo -e "${GREEN}✓ Running from correct directory${NC}"
echo ""

# Phase 1: Create directory structure
echo -e "${YELLOW}Phase 1: Creating directory structure...${NC}"

directories=(
    "data"
    "logs"
    "vector_store"
    "sessions"
    "model_cache"
    "document_processing"
    "query_handling"
    "query_processing"
    "retrieval"
    "response"
    "vector_management"
    "session_management"
)

for dir in "${directories[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo -e "${GREEN}  ✓ Created: $dir/${NC}"
    else
        echo -e "${BLUE}  • Exists: $dir/${NC}"
    fi
done

echo ""

# Create .gitignore
echo -e "${YELLOW}Creating .gitignore...${NC}"
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Environment
.env
.env.local
.env.production
*.local

# Data and Models
data/*.txt
data/*.pdf
data/*.docx
data/*.md
!data/README.md
vector_store/
sessions/
model_cache/
logs/*.log*

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Testing
.pytest_cache/
.coverage
htmlcov/

# Jupyter
.ipynb_checkpoints/

# spaCy models
en_core_web_*/
EOF
echo -e "${GREEN}✓ .gitignore created${NC}"
echo ""

# Create data README
echo -e "${YELLOW}Creating data/README.md...${NC}"
cat > data/README.md << 'EOF'
# Knowledge Base Data

Place your knowledge base files in this directory.

## Supported Formats

- **Text files:** `.txt`, `.md`
- **PDF documents:** `.pdf`
- **Word documents:** `.docx`, `.doc`
- **PowerPoint:** `.pptx`, `.ppt`
- **EPUB books:** `.epub`

## Usage

1. Add your documents to this folder
2. Run `python main.py`
3. The system will automatically index all supported files
4. Ask questions about your documents!

## Reindexing

If you add new documents or modify existing ones:
- In CLI: Type `reindex`
- The system will rebuild the index with your updated content

## Tips

- Organize files by topic/category if needed
- Use descriptive filenames
- Remove outdated content to keep results relevant
- Larger files may take longer to process
EOF
echo -e "${GREEN}✓ data/README.md created${NC}"
echo ""

# Phase 2: Copy files from SARA
echo -e "${YELLOW}Phase 2: Copying files from SARA...${NC}"
echo ""

# Copy core files
echo -e "${BLUE}Copying core files...${NC}"
core_files=(
    "main.py"
    "app.py"
    "api.py"
    "config.py"
    "resource_manager.py"
    "input_processing.py"
    "nlp_config_loader.py"
    "spacy_semantic_processor.py"
    "topics.yaml"
    "requirements.txt"
)

for file in "${core_files[@]}"; do
    if [ -f "$SARA_PATH/$file" ]; then
        cp "$SARA_PATH/$file" .
        echo -e "${GREEN}  ✓ Copied: $file${NC}"
    else
        echo -e "${RED}  ✗ Not found: $file${NC}"
    fi
done

# Copy and rename sara_types.py
if [ -f "$SARA_PATH/sara_types.py" ]; then
    cp "$SARA_PATH/sara_types.py" ragi_types.py
    echo -e "${GREEN}  ✓ Copied & renamed: sara_types.py → ragi_types.py${NC}"
else
    echo -e "${RED}  ✗ Not found: sara_types.py${NC}"
fi

echo ""

# Copy module directories
echo -e "${BLUE}Copying module directories...${NC}"
modules=(
    "document_processing"
    "query_handling"
    "query_processing"
    "retrieval"
    "response"
    "vector_management"
    "session_management"
)

for module in "${modules[@]}"; do
    if [ -d "$SARA_PATH/$module" ]; then
        cp -r "$SARA_PATH/$module"/* "$module/" 2>/dev/null || true
        echo -e "${GREEN}  ✓ Copied: $module/${NC}"
    else
        echo -e "${YELLOW}  ! Not found: $module/${NC}"
    fi
done

echo ""

# Phase 3: Global renames
echo -e "${YELLOW}Phase 3: Applying global renames...${NC}"
echo ""

echo -e "${BLUE}Renaming sara_types → ragi_types...${NC}"
find . -type f -name "*.py" -not -path "./.git/*" -exec sed -i '' 's/sara_types/ragi_types/g' {} + 2>/dev/null || true
echo -e "${GREEN}✓ Done${NC}"

echo -e "${BLUE}Updating logger names...${NC}"
find . -type f -name "*.py" -not -path "./.git/*" -exec sed -i '' 's/getLogger("Sara")/getLogger("RAGI")/g' {} + 2>/dev/null || true
find . -type f -name "*.py" -not -path "./.git/*" -exec sed -i '' 's/getLogger('\''Sara'\'')/getLogger('\''RAGI'\'')/g' {} + 2>/dev/null || true
echo -e "${GREEN}✓ Done${NC}"

echo -e "${BLUE}Renaming environment variables SARA_ → RAGI_...${NC}"
find . -type f -name "*.py" -not -path "./.git/*" -exec sed -i '' 's/SARA_/RAGI_/g' {} + 2>/dev/null || true
echo -e "${GREEN}✓ Done${NC}"

echo ""

# Create .env.template
echo -e "${YELLOW}Creating .env.template...${NC}"
cat > .env.template << 'EOF'
# RAGI Configuration Template
# Copy this to .env and customize

# Environment
RAGI_ENV=local

# Paths
RAGI_DATA_PATH=./data
RAGI_VECTOR_PATH=./vector_store

# Models
RAGI_EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
RAGI_LLM_MODEL=qwen2.5:3b-instruct
RAGI_RERANKER_MODEL=BAAI/bge-reranker-base
RAGI_SEMANTIC_MODEL=en_core_web_md

# Chunking
RAGI_CHUNK_SIZE=500
RAGI_CHUNK_OVERLAP=150

# Retrieval
RAGI_RETRIEVER_K=6
RAGI_SEARCH_TYPE=hybrid
RAGI_KEYWORD_RATIO=0.4

# FAQ Matching (optional)
RAGI_USE_FAQ_MATCHING=True
RAGI_FAQ_MATCH_WEIGHT=0.5
# RAGI_KB_CONTENT_TYPES=faq,kb_page,article

# Resources
RAGI_MAX_THREADS=4
RAGI_MAX_MEMORY=4G

# Logging
RAGI_LOG_LEVEL=INFO
RAGI_LOG_MAX_BYTES=1073741824
RAGI_LOG_BACKUP_COUNT=5

# Ollama
RAGI_OLLAMA_URL=http://localhost:11434

# Support Contact (customize for your org)
RAGI_SUPPORT_PHONE="+1-xxx-xxx-xxxx"
RAGI_SUPPORT_EMAIL="support@example.com"
RAGI_SUPPORT_LOCATION="Support Center"

# Language Detection
RAGI_LANGUAGE_DETECTION=True
RAGI_LANG_CONFIDENCE_THRESHOLD=0.65
RAGI_SUPPORTED_LANGUAGES=en

# Ambiguity Detection
RAGI_AMBIGUITY_DETECTION=True
RAGI_AMBIGUITY_THRESHOLD=0.7
EOF
echo -e "${GREEN}✓ .env.template created${NC}"
echo ""

# Summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Setup Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${GREEN}✓ Directory structure created${NC}"
echo -e "${GREEN}✓ Files copied from SARA${NC}"
echo -e "${GREEN}✓ Global renames applied${NC}"
echo -e "${GREEN}✓ Configuration templates created${NC}"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo -e "1. ${BLUE}Review and modify critical files:${NC}"
echo "   - config.py (lines 213-215, 354, 360-361, 435-453)"
echo "   - document_processing/parsers.py (complete rewrite)"
echo "   - retrieval/faq_matcher.py (line 74)"
echo "   - topics.yaml (rewrite with generic topics)"
echo ""
echo -e "2. ${BLUE}Install dependencies:${NC}"
echo "   conda activate RAGI"
echo "   pip install -r requirements.txt"
echo "   python -m spacy download en_core_web_md"
echo "   ollama pull qwen2.5:3b-instruct"
echo ""
echo -e "3. ${BLUE}Test installation:${NC}"
echo "   python -c \"from config import config; print('Config OK')\""
echo "   python -c \"from ragi_types import QueryType; print('Types OK')\""
echo ""
echo -e "4. ${BLUE}Review documentation:${NC}"
echo "   - RAGI_MIGRATION_PLAN.md (full migration guide)"
echo "   - RAGI_HANDOVER_DOCUMENT.md (project status)"
echo "   - RAGI_QUICK_REFERENCE.md (quick commands)"
echo ""
echo -e "${YELLOW}⚠️  Manual modifications still required!${NC}"
echo "   See RAGI_MIGRATION_PLAN.md Phase 5 for details"
echo ""
