#!/bin/bash
# Local development startup script for FallGuard

echo "=================================================="
echo "FallGuard - Local Development Setup"
echo "=================================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed"
    echo "Please install Python 3.10 or higher"
    exit 1
fi

echo "✅ Python found: $(python3 --version)"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo ""
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo ""
echo "📥 Installing dependencies..."
pip install -r requirements.txt -q

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Create necessary directories
echo ""
echo "📁 Creating necessary directories..."
mkdir -p data
mkdir -p models
mkdir -p uploads
echo "✅ Directories created"

# Copy .env if it doesn't exist
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        echo ""
        echo "📋 Creating .env from .env.example..."
        cp .env.example .env
        echo "⚠️  Please edit .env with your configuration"
    fi
fi

echo ""
echo "=================================================="
echo "✅ Setup complete!"
echo "=================================================="
echo ""
echo "To start the application, run:"
echo "  python main.py"
echo ""
echo "Or with Gunicorn (like Render):"
echo "  gunicorn --timeout 120 --workers 1 main:app"
echo ""
echo "Access at: http://localhost:5000"
echo "=================================================="
