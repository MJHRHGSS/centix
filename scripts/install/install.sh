#!/bin/bash
set -e
echo "Installing cx..."
cd "$(dirname "$(readlink -f "$0")")"
cd ../..
echo "Compiling..."
scripts/build.sh
echo "Moving files..."
sudo cp out/cx /usr/local/bin/cx
sudo chmod +x /usr/local/bin/cx
echo "Done!"
