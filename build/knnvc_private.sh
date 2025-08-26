#!/bin/bash

DOWNLOAD_URL="https://github.com/carlosfranzreb/private_knnvc/releases/download/v1.0.0/checkpoints.zip"
OUTPUT_FILENAME="knnvc_private_checkpoints.zip"

# Check that you are in  the correct directory
dir_name="spane"
if [ ! -d "$dir_name" ]; then
    echo "Error: You should run this script from the parent directory of SpAnE."
    exit 1
fi

# Download and unzip checkpoints
curl -L -o "$OUTPUT_FILENAME" "$DOWNLOAD_URL"
unzip "$OUTPUT_FILENAME"
