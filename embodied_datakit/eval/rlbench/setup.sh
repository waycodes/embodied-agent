#!/usr/bin/env bash
# RLBench headless setup script
# Installs CoppeliaSim and RLBench for headless evaluation

set -e

COPPELIASIM_VERSION="4.1.0"
INSTALL_DIR="${INSTALL_DIR:-$HOME/.local/rlbench}"

echo "=== RLBench Headless Setup ==="
echo "Install directory: $INSTALL_DIR"

mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# Check for required dependencies
if ! command -v xvfb-run &> /dev/null; then
    echo "Installing xvfb for headless rendering..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y xvfb
    else
        echo "Warning: xvfb not found. Install manually for headless mode."
    fi
fi

# Download CoppeliaSim if not present
if [ ! -d "CoppeliaSim" ]; then
    echo "Downloading CoppeliaSim $COPPELIASIM_VERSION..."
    COPPELIA_URL="https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V${COPPELIASIM_VERSION}_Ubuntu20_04.tar.xz"
    curl -L "$COPPELIA_URL" -o coppeliasim.tar.xz
    tar -xf coppeliasim.tar.xz
    mv CoppeliaSim_Edu_V${COPPELIASIM_VERSION}_Ubuntu20_04 CoppeliaSim
    rm coppeliasim.tar.xz
fi

# Set environment variables
export COPPELIASIM_ROOT="$INSTALL_DIR/CoppeliaSim"
export LD_LIBRARY_PATH="$COPPELIASIM_ROOT:$LD_LIBRARY_PATH"
export QT_QPA_PLATFORM="offscreen"

# Install PyRep
if ! python -c "import pyrep" 2>/dev/null; then
    echo "Installing PyRep..."
    pip install git+https://github.com/stepjam/PyRep.git
fi

# Install RLBench
if ! python -c "import rlbench" 2>/dev/null; then
    echo "Installing RLBench..."
    pip install git+https://github.com/stepjam/RLBench.git
fi

# Create activation script
cat > "$INSTALL_DIR/activate.sh" << 'EOF'
export COPPELIASIM_ROOT="$HOME/.local/rlbench/CoppeliaSim"
export LD_LIBRARY_PATH="$COPPELIASIM_ROOT:$LD_LIBRARY_PATH"
export QT_QPA_PLATFORM="offscreen"
echo "RLBench environment activated"
EOF

echo ""
echo "=== Setup Complete ==="
echo "To activate: source $INSTALL_DIR/activate.sh"
echo "To run headless: xvfb-run -a python your_script.py"
