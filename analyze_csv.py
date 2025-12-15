import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing_utils import quick_classify

# Set style for plots
sns.set_theme(style="whitegrid")
# Ensure matplotlib can display Vietnamese characters if supported font is found, 
# otherwise fallback to default. 
# In a standard env, this might be tricky, so we stick to standard fonts but write strings in Vietnamese.

def load_data(filepath):
    """Loads CSV data."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Không tìm thấy file: {filepath}")
    
    # Check if empty
    if os.path.getsize(filepath) == 0:
         raise ValueError(f"File rỗng: {filepath}")

    print(f"Đang tải dữ liệu từ: {filepath} ...")
    # Low memory false to handle mixed types if any, similar to training script
    df = pd.read_csv(filepath, low_memory=False)
    return df

def analyze_data(df):
    """Calculates statistics and metrics."""
    stats = {}
    
    # 1. Label Mapping
    if 'Label' in df.columns:
        df['Mapped_Label'] = df['Label'].apply(quick_classify)
        
        # Raw Label Counts
        stats['raw_label_counts'] = df['Label'].value_counts().head(20).to_dict()
        
        # Mapped Label Counts
        stats['mapped_label_counts'] = df['Mapped_Label'].value_counts().to_dict()
        stats['mapped_label_pct'] = df['Mapped_Label'].value_counts(normalize=True).to_dict()
    else:
        stats['raw_label_counts'] = {}
        stats['mapped_label_counts'] = {}
        print("CẢNH BÁO: Không tìm thấy cột 'Label'.")

    # 2. Numerical Features Stats
    num_cols = ['Dur', 'TotPkts', 'TotBytes', 'SrcBytes']
    stats['numerical'] = {}
    
    for col in num_cols:
        if col in df.columns:
            # Force numeric
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            desc = df[col].describe()
            stats['numerical'][col] = {
                'mean': desc['mean'],
                'std': desc['std'],
                'min': desc['min'],
                'max': desc['max'],
                'median': desc['50%']
            }
        else:
            stats['numerical'][col] = None

    # 3. Categorical Counts
    cat_cols = ['Proto', 'State', 'Dir']
    stats['categorical'] = {}
    for col in cat_cols:
        if col in df.columns:
            stats['categorical'][col] = df[col].value_counts().head(10).to_dict()
        else:
            stats['categorical'][col] = {}
            
    # 4. Missing Values
    stats['missing_values'] = df.isnull().sum().to_dict()
    
    return stats, df

def generate_report(stats, df, output_dir, filename_base):
    """Generates text report and plots."""
    
    # --- 1. Text Report ---
    report_path = os.path.join(output_dir, f"{filename_base}_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"BÁO CÁO PHÂN TÍCH DỮ LIỆU: {filename_base}\n")
        f.write("="*60 + "\n\n")
        
        f.write("1. TỔNG QUAN\n")
        f.write(f"   - Tổng số dòng: {len(df)}\n")
        f.write(f"   - Số cột: {len(df.columns)}\n")
        f.write("\n")
        
        f.write("2. PHÂN BỐ NHÃN (Lớp Mapped)\n")
        if stats['mapped_label_counts']:
            for label, count in stats['mapped_label_counts'].items():
                pct = stats['mapped_label_pct'].get(label, 0) * 100
                f.write(f"   - {label}: {count} ({pct:.2f}%)\n")
        else:
            f.write("   (Không có dữ liệu nhãn)\n")
        f.write("\n")
        
        f.write("3. TOP 20 NHÃN GỐC (Raw Labels)\n")
        if stats['raw_label_counts']:
            for label, count in stats['raw_label_counts'].items():
                f.write(f"   - {label}: {count}\n")
        f.write("\n")
        
        f.write("4. THỐNG KÊ SỐ HỌC (Numerical Stats)\n")
        for col, data in stats['numerical'].items():
            if data:
                f.write(f"   * {col}:\n")
                f.write(f"     - Trung bình (Mean): {data['mean']:.2f}\n")
                f.write(f"     - Trung vị (Median): {data['median']:.2f}\n")
                f.write(f"     - Lớn nhất (Max):    {data['max']:.2f}\n")
                f.write(f"     - Độ lệch chuẩn (Std): {data['std']:.2f}\n")
            else:
                f.write(f"   * {col}: Không tìm thấy\n")
        f.write("\n")

        f.write("5. THÔNG TIN PHÂN LOẠI (Categorical Info)\n")
        for col, data in stats['categorical'].items():
            f.write(f"   * Top 10 {col}:\n")
            for k, v in data.items():
                f.write(f"     - {k}: {v}\n")
        f.write("\n")

        f.write("6. GIÁ TRỊ THIẾU (Missing Values)\n")
        for col, count in stats['missing_values'].items():
            if count > 0:
                f.write(f"   - {col}: {count}\n")
    
    print(f"Đã lưu báo cáo văn bản tại: {report_path}")

    # --- 2. Visualization ---
    image_path = os.path.join(output_dir, f"{filename_base}_visuals.png")
    
    # Setup Figure: 2 Rows, 3 Columns
    fig = plt.figure(figsize=(20, 12), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    # A. Class Distribution (Mapped)
    ax1 = fig.add_subplot(gs[0, 0])
    if 'Mapped_Label' in df.columns:
        sns.countplot(x='Mapped_Label', data=df, ax=ax1, palette='viridis', hue='Mapped_Label', legend=False)
        ax1.set_title("Phân bố Lớp (Mapped Classes)", fontsize=14)
        ax1.set_ylabel("Số lượng")
        ax1.set_xlabel("Lớp")
    else:
        ax1.text(0.5, 0.5, "Không có dữ liệu nhãn", ha='center')

    # B. Numerical Distributions (Boxplots Log Scale)
    ax2 = fig.add_subplot(gs[0, 1])
    # Melting for easier plotting of multiple features
    num_cols = ['Dur', 'TotPkts', 'TotBytes', 'SrcBytes']
    # Filter valid columns
    valid_num = [c for c in num_cols if c in df.columns]
    
    if valid_num:
        # We take a sample if data is too huge to plot quickly? No, boxplot is fast enough usually.
        # But we need log scale because Bytes can be huge.
        df_melt = df[valid_num].melt(var_name='Feature', value_name='Value')
        # Log transformation for display (handling 0)
        df_melt['LogValue'] = np.log1p(df_melt['Value'])
        
        sns.boxplot(x='Feature', y='LogValue', data=df_melt, ax=ax2, hue='Feature', palette="Set2")
        ax2.set_title("Phân bố Đặc trưng Số (Log Scale)", fontsize=14)
        ax2.set_ylabel("Log(Giá trị + 1)")
    else:
        ax2.text(0.5, 0.5, "Không có dữ liệu số", ha='center')

    # C. Top Raw Labels (Horizontal Bar)
    ax3 = fig.add_subplot(gs[0, 2])
    if 'Label' in df.columns:
        top_labels = df['Label'].value_counts().head(10)
        sns.barplot(y=top_labels.index, x=top_labels.values, ax=ax3, palette="magma", hue=top_labels.index, legend=False)
        ax3.set_title("Top 10 Nhãn Gốc (Raw Labels)", fontsize=14)
        ax3.set_xlabel("Số lượng")
    else:
        ax3.text(0.5, 0.5, "Không có dữ liệu nhãn", ha='center')

    # D. Protocol Distribution
    ax4 = fig.add_subplot(gs[1, 0])
    if 'Proto' in df.columns:
        top_proto = df['Proto'].value_counts().head(10)
        sns.barplot(x=top_proto.index, y=top_proto.values, ax=ax4, palette="Blues_d", hue=top_proto.index, legend=False)
        ax4.set_title("Top Giao thức (Protocol)", fontsize=14)
        ax4.set_ylabel("Số lượng")
    else:
        ax4.text(0.5, 0.5, "Không có Proto", ha='center')

    # E. State Distribution
    ax5 = fig.add_subplot(gs[1, 1])
    if 'State' in df.columns:
        top_state = df['State'].value_counts().head(10)
        sns.barplot(x=top_state.index, y=top_state.values, ax=ax5, palette="Reds_d", hue=top_state.index, legend=False)
        ax5.set_title("Top Trạng thái (State)", fontsize=14)
        ax5.set_ylabel("Số lượng")
        ax5.tick_params(axis='x', rotation=45)
    else:
        ax5.text(0.5, 0.5, "Không có State", ha='center')

    # F. Feature by Class (e.g., TotBytes per Class)
    ax6 = fig.add_subplot(gs[1, 2])
    if 'Mapped_Label' in df.columns and 'TotBytes' in df.columns:
         # Use LogBytes
         df['LogTotBytes'] = np.log1p(df['TotBytes'])
         sns.boxplot(x='Mapped_Label', y='LogTotBytes', data=df, ax=ax6, palette="coolwarm", hue='Mapped_Label', legend=False)
         ax6.set_title("TotBytes theo Lớp (Log Scale)", fontsize=14)
         ax6.set_ylabel("Log(TotBytes)")
    else:
        ax6.text(0.5, 0.5, "Thiếu dữ liệu để vẽ biểu đồ", ha='center')

    # Save
    plt.suptitle(f"Dashboard Phân Tích: {filename_base}", fontsize=20)
    plt.savefig(image_path)
    plt.close()
    
    print(f"Đã lưu biểu đồ tại: {image_path}")


def main():
    parser = argparse.ArgumentParser(description="Script phân tích file CSV CTU-13")
    parser.add_argument("--file", type=str, required=True, help="Đường dẫn đến file .csv")
    parser.add_argument("--outdir", type=str, default="analysis_reports", help="Thư mục lưu kết quả")
    
    args = parser.parse_args()
    
    # Create output dir
    os.makedirs(args.outdir, exist_ok=True)
    
    # Filename base
    base_name = os.path.splitext(os.path.basename(args.file))[0]
    # To avoid overwriting if same filename exists in different folders, maybe prepend parent folder?
    # e.g. 10_10.csv
    parent_dir = os.path.basename(os.path.dirname(args.file))
    if parent_dir.isdigit():
        base_name = f"{parent_dir}_{base_name}"
    
    try:
        df = load_data(args.file)
        stats, df_processed = analyze_data(df)
        generate_report(stats, df_processed, args.outdir, base_name)
        print("Hoàn tất!")
    except Exception as e:
        print(f"LỖI: {e}")

if __name__ == "__main__":
    main()
