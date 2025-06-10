# note i havent tested this
#!/bin/bash

# Script to move files to destinations specified within their content

# Directory to scan for files (current directory by default)
SOURCE_DIR="./"

# Function to extract destination path from file content
extract_destination() {
    local file="$1"
    # Look for path patterns in the file - adjust this grep pattern based on 
    # how the paths are formatted in your files
    dest_path=$(grep -o '/[a-zA-Z0-9_/.-]\+' "$file" | head -n1)
    
    # If no path found, try another common pattern
    if [ -z "$dest_path" ]; then
        dest_path=$(grep -o 'path:[ ]*[a-zA-Z0-9_/.-]\+' "$file" | sed 's/path:[ ]*//' | head -n1)
    fi
    
    echo "$dest_path"
}

# Process files
count=0
for file in "$SOURCE_DIR"/*; do
    # Skip directories
    [ -d "$file" ] && continue
    
    # Skip the script itself
    [ "$(basename "$file")" = "$(basename "$0")" ] && continue
    
    # Extract destination from file content
    destination=$(extract_destination "$file")
    
    # Check if destination was found
    if [ -n "$destination" ]; then
        # Create destination directory if it doesn't exist
        mkdir -p "$(dirname "$destination")"
        
        # Move the file
        echo "Moving $file to $destination"
        mv "$file" "$destination"
        
        count=$((count + 1))
        
        # Stop after processing 4 files
        [ "$count" -eq 4 ] && break
    fi
done

echo "Moved $count files to their destinations."