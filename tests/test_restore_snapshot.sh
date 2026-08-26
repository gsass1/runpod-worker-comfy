#!/usr/bin/env bash

# Set up error handling
set -e

# Store the path to the script we want to test
SCRIPT_TO_TEST="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/src/restore_snapshot.sh"

# Ensure the script exists and is executable
if [ ! -f "$SCRIPT_TO_TEST" ]; then
    echo "Error: Script not found at $SCRIPT_TO_TEST"
    exit 1
fi
chmod +x "$SCRIPT_TO_TEST"

# Create test directory
TEST_DIR=$(mktemp -d)
trap 'rm -rf "$TEST_DIR"' EXIT
cd "$TEST_DIR"

# Create a minimal mock snapshot file
cat > test_restore_snapshot_temporary.json << 'EOF'
{
  "comfyui": "test-hash",
  "git_custom_nodes": {},
  "file_custom_nodes": [],
  "pips": {}
}
EOF

# Create a mock comfy command that simulates the real comfy behavior
cat > comfy << 'EOF'
#!/bin/bash
if [[ "$1" == "--workspace" && "$2" == "/comfyui" && "$3" == "node" && "$4" == "restore-snapshot" ]]; then
    # Verify the snapshot file exists
    if [[ ! -f "$5" ]]; then
        echo "Error: Snapshot file not found"
        exit 1
    fi
    touch "$COMFY_CALL_MARKER"
    echo "Mock: Restored snapshot from $5"
    exit 0
fi
echo "Error: Unexpected comfy arguments: $*"
exit 1
EOF

chmod +x comfy
export PATH="$TEST_DIR:$PATH"
export COMFY_CALL_MARKER="$TEST_DIR/comfy-called"

# Run the actual restore_snapshot script
echo "Testing snapshot restoration..."
echo "Script location: $SCRIPT_TO_TEST"
SNAPSHOT_FILE="$TEST_DIR/test_restore_snapshot_temporary.json" "$SCRIPT_TO_TEST"

if [ ! -f "$COMFY_CALL_MARKER" ]; then
    echo "Error: Mock comfy command was not called"
    exit 1
fi

echo "✅ Test passed: Snapshot restoration script executed successfully"
