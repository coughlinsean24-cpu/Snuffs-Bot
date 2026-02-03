#!/bin/bash
# Setup script for Background Learner auto-start service
# This makes the AI learn continuously during market hours

echo "=== Setting up Background Learner Service ==="
echo ""

# Create user systemd directory
mkdir -p ~/.config/systemd/user

# Copy service file
cp "/home/coughlinsean24/Snuffs Bot/scripts/snuffs-learner.service" ~/.config/systemd/user/

# Reload systemd
systemctl --user daemon-reload

# Enable service (starts on boot)
systemctl --user enable snuffs-learner.service

# Start service now
systemctl --user start snuffs-learner.service

# Enable lingering (allows user services to run even when not logged in)
sudo loginctl enable-linger $USER 2>/dev/null || echo "Note: Run 'sudo loginctl enable-linger $USER' to keep service running when logged out"

echo ""
echo "✅ Background Learner service installed!"
echo ""
echo "Commands:"
echo "  Status:  systemctl --user status snuffs-learner"
echo "  Stop:    systemctl --user stop snuffs-learner"
echo "  Start:   systemctl --user start snuffs-learner"
echo "  Logs:    journalctl --user -u snuffs-learner -f"
echo ""
echo "The AI will now learn automatically during market hours (9:30 AM - 4:15 PM EST)"
echo "even when the trading bot is OFF!"
