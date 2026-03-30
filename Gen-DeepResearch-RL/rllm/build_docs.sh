#!/bin/bash

# Install mkdocs and required plugins if not already installed
echo "Installing documentation dependencies with uv..."
uv pip install -e .

# Ensure the rllm package is available for import
export PYTHONPATH="../:$PYTHONPATH"

# Change to docs directory
cd "$(dirname "$0")"

# Build the documentation
echo "Building documentation..."
mkdocs build

# Serve the documentation (if requested)
if [ "$1" == "serve" ]; then
    echo "Starting documentation server..."
    mkdocs serve
fi

echo "Documentation built successfully!"
echo "To serve the documentation, run: cd docs && mkdocs serve"
echo "To view the built documentation, open: docs/site/index.html" 