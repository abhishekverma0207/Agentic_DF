#!/bin/bash
# =============================================================================
# Databricks Cluster Init Script for FEU-Agentic-Forecasting
# =============================================================================
# This script runs automatically when the cluster starts.
# It installs all Python dependencies from requirements.txt.
#
# SETUP:
# 1. Upload this script to DBFS: dbfs:/FileStore/init-scripts/feu_init.sh
# 2. Configure cluster: Compute → Edit → Advanced → Init Scripts → Add path
#
# ALTERNATIVE: If using Databricks Repos, the script can also be referenced
# directly from the workspace path.
# =============================================================================

echo "========================================"
echo "FEU-Agentic-Forecasting: Installing Dependencies"
echo "========================================"
echo "Start time: $(date)"
echo "Python path: $(which python)"
echo "Pip path: $(which pip)"

# Define possible repo locations (adjust as needed for your setup)
REPO_PATHS=(
    "/Workspace/Repos/debonil.chowdhry@unilever.com/FEUAgenticForecasting"
    "/Workspace/Repos/${DB_USER_EMAIL:-user}/FEUAgenticForecasting"
    "/Workspace/Repos/FEUAgenticForecasting"
    "/Workspace/Users/${DB_USER_EMAIL:-user}/FEUAgenticForecasting"
)

# Find the requirements.txt file
REQUIREMENTS_FILE=""
for path in "${REPO_PATHS[@]}"; do
    echo "Checking: $path/requirements.txt"
    if [ -f "$path/requirements.txt" ]; then
        REQUIREMENTS_FILE="$path/requirements.txt"
        echo ">>> FOUND requirements.txt at: $REQUIREMENTS_FILE"
        break
    fi
done

# Install dependencies
if [ -n "$REQUIREMENTS_FILE" ]; then
    echo ""
    echo "========================================"
    echo "Installing Python dependencies from requirements.txt..."
    echo "========================================"

    # Upgrade pip first
    /databricks/python/bin/pip install --upgrade pip --quiet

    # Install all requirements (verbose to see progress)
    /databricks/python/bin/pip install -r "$REQUIREMENTS_FILE" --no-warn-script-location 2>&1

    INSTALL_STATUS=$?

    if [ $INSTALL_STATUS -eq 0 ]; then
        echo ""
        echo "========================================"
        echo "SUCCESS: Dependencies installed successfully"
        echo "========================================"
    else
        echo ""
        echo "========================================"
        echo "WARNING: Some dependencies may have failed (exit code: $INSTALL_STATUS)"
        echo "========================================"
    fi

    # Verify critical packages are installed
    echo ""
    echo "Verifying critical packages..."
    /databricks/python/bin/pip show crewai 2>/dev/null && echo ">>> crewai: INSTALLED" || echo ">>> crewai: NOT FOUND"
    /databricks/python/bin/pip show litellm 2>/dev/null && echo ">>> litellm: INSTALLED" || echo ">>> litellm: NOT FOUND"
    /databricks/python/bin/pip show pandas 2>/dev/null && echo ">>> pandas: INSTALLED" || echo ">>> pandas: NOT FOUND"

else
    echo ""
    echo "========================================"
    echo "WARNING: requirements.txt not found!"
    echo "========================================"
    echo "Searched paths:"
    for path in "${REPO_PATHS[@]}"; do
        echo "  - $path/requirements.txt"
    done
    echo ""
    echo "Installing critical packages directly..."

    # Fallback: Install critical packages directly
    /databricks/python/bin/pip install crewai>=1.7.2 litellm>=1.34.10 boto3>=1.34.50 pydantic>=2.0 --no-warn-script-location 2>&1

    echo "Fallback installation completed"
fi

echo ""
echo "========================================"
echo "Init script completed at: $(date)"
echo "========================================"
