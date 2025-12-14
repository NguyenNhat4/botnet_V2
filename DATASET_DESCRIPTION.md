# Mô tả Dữ liệu CTU-13 (File 10.csv)

File này chứa dữ liệu **Network Flow** (luồng mạng), được sử dụng để phát hiện Botnet. Dưới đây là giải thích chi tiết về ý nghĩa của từng cột trong tập dữ liệu.

## 1. Thông tin Định danh (Identifiers)
Các cột này xác định nguồn gốc và đích đến của kết nối.

| Cột | Tên đầy đủ | Ý nghĩa |
|-----|------------|---------|
| **StartTime** | Start Time | Thời gian bắt đầu của luồng mạng. |
| **Proto** | Protocol | Giao thức vận chuyển (ví dụ: `tcp`, `udp`, `icmp`). Botnet thường dùng UDP cho DDoS hoặc TCP cho điều khiển. |
| **SrcAddr** | Source Address | Địa chỉ IP của thiết bị bắt đầu kết nối (Nguồn). |
| **Sport** | Source Port | Cổng (Port) bên phía nguồn. |
| **DstAddr** | Destination Address | Địa chỉ IP của thiết bị nhận kết nối (Đích). |
| **Dport** | Destination Port | Cổng (Port) bên phía đích (ví dụ: 80 cho Web, 443 cho HTTPS). |

## 2. Thông số Đo lường (Metrics)
Các chỉ số định lượng về kích thước và thời gian của luồng.

| Cột | Ý nghĩa | Suy luận |
|-----|---------|----------|
| **Dur** | Thời gian tồn tại của luồng (giây). | Thời gian quá ngắn (< 1s) có thể là quét cổng; quá dài có thể là kênh C&C liên tục. |
| **TotPkts** | Tổng số gói tin (Packets). | Số lượng gói tin trao đổi trong suốt phiên kết nối. |
| **TotBytes** | Tổng dung lượng (Bytes). | Tổng kích thước dữ liệu trao đổi (cả gửi và nhận). |
| **SrcBytes** | Dung lượng gửi từ nguồn. | Nếu `SrcBytes` chiếm phần lớn `TotBytes`, máy nguồn chủ yếu đang gửi dữ liệu đi (có thể là tấn công hoặc exfiltration). |

## 3. Thông tin Hành vi & Trạng thái (Behavior)
Mô tả cách thức và trạng thái của kết nối.

- **Dir** (Direction): Hướng của luồng dữ liệu.
  - `<->`: Hai chiều (Trao đổi bình thường).
  - `->`: Một chiều (Chỉ gửi hoặc chỉ nhận).
  - `<?>`: Không xác định.

- **State** (Trạng thái): Trạng thái kết nối (đặc biệt quan trọng với TCP).
  - `CON`: Connected (Kết nối thành công).
  - `URP`: Unreached Port (Thường thấy khi quét cổng/lỗi).
  - Các cờ TCP: `FPA`, `S`, `R`... (Phản ánh quá trình bắt tay hoặc ngắt kết nối).

- **sTos** / **dTos**: Type of Service. Thường dùng cho QoS (Quality of Service), ít có giá trị phân loại botnet trong tập này.

## 4. Nhãn Dữ liệu (Target Label)
Cột quan trọng nhất để huấn luyện mô hình Machine Learning.

- **Label**: Chứa thông tin chi tiết về loại luồng.
  - Ví dụ: `flow=Background-Established-cmpgw-CVUT`
  - **Xử lý (Preprocessing)**: Thông thường cột này sẽ được gộp nhóm thành 3 lớp chính:
    1. **Botnet**: Lưu lượng độc hại từ các máy bị nhiễm.
    2. **C&C**: Lưu lượng giao tiếp với máy chủ điều khiển (Command & Control).
    3. **Normal**: Lưu lượng sạch/bình thường (Background).

