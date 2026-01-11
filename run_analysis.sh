#!/bin/bash

# Directory containing dataset
DATA_DIR="CTU-13-Dataset"
OUTPUT_DIR="analysis_reports"

echo "Bắt đầu quét và phân tích dữ liệu trong $DATA_DIR..."

# Find all .csv files (recursively)
# We use 'find' to handle different depth levels if needed
find "$DATA_DIR" -name "*.csv" | sort | while read -r file; do
    echo "--------------------------------------------------"
    echo "Đang xử lý: $file"
    python analyze_csv.py --file "$file" --outdir "$OUTPUT_DIR"
done

echo "--------------------------------------------------"
echo "Hoàn tất! Kiểm tra kết quả trong thư mục '$OUTPUT_DIR'."
