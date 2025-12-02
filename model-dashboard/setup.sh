#!/bin/bash

echo "============================================"
echo "Model Comparison Dashboard - Setup Script"
echo "============================================"
echo ""

# Install dependencies
echo "📦 Installing dependencies..."
npm install

# Create directories
echo "📁 Creating directories..."
mkdir -p public/logos
mkdir -p public/data

# Check for logo files in parent directory
echo "🖼️  Looking for logo files..."
if [ -d "../models_results" ]; then
    echo "Found models_results folder, copying logos..."
    
    for logo in openai.png claude.png deepseek.png qwen.png gemini.png trained.png; do
        if [ -f "../models_results/$logo" ]; then
            cp "../models_results/$logo" public/logos/
            echo "  ✓ Copied $logo"
        else
            echo "  ⚠️  Not found: $logo"
        fi
    done
    
    echo ""
    echo "📊 Copying evaluation data..."
    cp ../models_results/eval_results_*.json public/data/ 2>/dev/null || echo "  ⚠️  No eval_results files found"
else
    echo "  ⚠️  models_results folder not found"
    echo "  Please copy logo files manually to public/logos/"
fi

echo ""
echo "============================================"
echo "✅ Setup complete!"
echo "============================================"
echo ""
echo "To start the dashboard:"
echo "  npm start"
echo ""
echo "The app will open at http://localhost:3000"
echo ""
