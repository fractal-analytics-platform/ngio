#!/bin/bash
# filepath: download_zarr_datasets.sh

# Default download directory
DOWNLOAD_DIR="data"

# Create download directory if it doesn't exist
mkdir -p "$DOWNLOAD_DIR"

# Function to check MD5 hash
check_md5() {
    local file="$1"
    local expected="$2"
    
    if [[ "$(uname)" == "Darwin" ]]; then
        # macOS
        actual=$(md5 -q "$file")
    else
        # Linux/Ubuntu
        actual=$(md5sum "$file" | awk '{print $1}')
    fi

    echo "Expected: $expected"
    echo "Actual: $actual"
    
    if [[ "$actual" == "$expected" ]]; then
        return 0  # Success
    else
        return 1  # Failure
    fi
}

# Function to download a file
download_file() {
    local url="$1"
    local output="$2"
    
    echo "Downloading $url to $output..."
    
    if command -v curl &> /dev/null; then
        curl -L -o "$output" "$url"
    elif command -v wget &> /dev/null; then
        wget -O "$output" "$url"
    else
        echo "Error: Neither curl nor wget is available. Please install one of them."
        exit 1
    fi
}

# Function to process a dataset
process_dataset() {
    local filename="$1"
    local url="$2"
    local expected_hash="$3"
    
    local file_path="$DOWNLOAD_DIR/$filename"
    
    echo "Processing $filename..."
    
    # Check if file exists and has the correct hash
    if [[ -f "$file_path" ]] && check_md5 "$file_path" "$expected_hash"; then
        echo "File exists and has the correct hash."
    else
        # File doesn't exist or has incorrect hash
        if [[ -f "$file_path" ]]; then
            echo "File exists but has incorrect hash. Redownloading..."
        else
            echo "File doesn't exist. Downloading..."
        fi
        
        download_file "$url" "$file_path"
        
        # Verify the downloaded file
        if check_md5 "$file_path" "$expected_hash"; then
            echo "Download successful and hash verified."
        else
            echo "Error: Downloaded file has incorrect hash."
            return 1
        fi
    fi
    
    echo "File is ready at $file_path"
    return 0
}

# Process the CardioMyocyte dataset
process_dataset "20200812-CardiomyocyteDifferentiation14-Cycle1.zarr.zip" \
                "https://zenodo.org/records/13305156/files/20200812-CardiomyocyteDifferentiation14-Cycle1.zarr.zip" \
                "efc21fe8d4ea3abab76226d8c166452c"

process_dataset "20200812-CardiomyocyteDifferentiation14-Cycle1_mip.zarr.zip" \
                "https://zenodo.org/records/13305316/files/20200812-CardiomyocyteDifferentiation14-Cycle1_mip.zarr.zip" \
                "3ed3ea898e0ed42d397da2e1dbe40750"
# To add more datasets, add more calls to process_dataset like this:
# process_dataset "filename.zip" "download_url" "expected_md5_hash"

echo "All datasets processed."