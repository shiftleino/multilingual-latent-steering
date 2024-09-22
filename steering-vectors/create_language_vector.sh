#!/bin/bash

DATA_DIR="floresp-v2.0-rc.3"
ZIP_URL="https://github.com/openlanguagedata/flores/releases/download/v2.0-rc.3/floresp-v2.0-rc.3.zip"

# Get the data
if [ -d "$DATA_DIR" ]; then
    echo "$DATA_DIR already exists. Skipping download."
else
    apt-get update
    apt-get install -y wget unzip

    wget $ZIP_URL
    unzip -P 'multilingual machine translation' $DATA_DIR.zip # password from the flores repo
    if [ $? -ne 0 ]; then
        echo "Failed to download the zip file."
        exit 1
    fi
fi

# Run the script to create the control direction
pip install -r requirements.txt
python main.py language-control ./model $DATA_DIR/dev/dev.fin_Latn $DATA_DIR/dev/dev.eng_Latn 997

if [ $? -ne 0 ]; then
    echo "Failed to create the control direction."
    exit 1
fi