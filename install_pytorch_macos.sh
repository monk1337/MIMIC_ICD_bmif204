#!/bin/bash

# Install PyTorch for macOS
# Detects Apple Silicon (M1/M2/M3) vs Intel and installs accordingly

echo "=========================================="
echo "Installing PyTorch for macOS"
echo "=========================================="
echo ""

# Detect architecture
ARCH=$(uname -m)

if [ "$ARCH" = "arm64" ]; then
    echo "✓ Detected Apple Silicon (M1/M2/M3)"
    echo "  Installing PyTorch with MPS (Metal Performance Shaders) support"
    echo ""
    
    # Install PyTorch with MPS support for Apple Silicon
    pip3 install torch torchvision torchaudio
    
elif [ "$ARCH" = "x86_64" ]; then
    echo "✓ Detected Intel Mac"
    echo "  Installing PyTorch CPU-only version"
    echo ""
    
    # Install PyTorch CPU-only for Intel Macs
    pip3 install torch torchvision torchaudio
else
    echo "✗ Unknown architecture: $ARCH"
    exit 1
fi

echo ""
echo "=========================================="
echo "Testing PyTorch Installation"
echo "=========================================="
echo ""

python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CPU available: {torch.cuda.is_available() or True}')

# Check for MPS (Apple Silicon GPU) support
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print(f'MPS (Apple GPU) available: True')
    print('  You can use: --device mps for GPU acceleration')
else:
    print(f'MPS (Apple GPU) available: False')
    print('  You will use: --device cpu for training')
"

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Note for macOS:"
echo "  - CUDA is not available on macOS"
echo "  - Use CPU mode: remove --gpu flag from commands"
echo "  - If you have Apple Silicon (M1/M2/M3):"
echo "    You can use MPS for GPU acceleration"
echo ""
echo "Updated command for macOS:"
echo "  python3 cnn/train_full_codes.py \\"
echo "    --model conv_attn \\"
echo "    --n-epochs 50 \\"
echo "    --batch-size 16 \\"
echo "    --lr 0.001 \\"
echo "    --dropout 0.5 \\"
echo "    --patience 5 \\"
echo "    --criterion f1_micro \\"
echo "    --seed 42"
echo "  (Remove --gpu flag)"
echo ""
