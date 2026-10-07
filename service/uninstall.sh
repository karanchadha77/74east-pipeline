#!/bin/bash
# Uninstall 74East Pipeline LaunchAgent

PLIST_NAME="com.74east-pipeline.dashboard"
PLIST_PATH="$HOME/Library/LaunchAgents/${PLIST_NAME}.plist"

if [ -f "$PLIST_PATH" ]; then
    echo "Stopping 74East Pipeline..."
    launchctl unload "$PLIST_PATH" 2>/dev/null || true
    rm -f "$PLIST_PATH"
    echo "74East Pipeline LaunchAgent removed."
else
    echo "74East Pipeline LaunchAgent not found."
fi
