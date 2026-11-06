# Kết nối Google Colab bằng MCP

## Trạng thái xác nhận ngày 31/10/2026

Đã ghép nối thật và gọi thành công get_cells/add_code_cell/run_code_cell qua MCP. Cell chẩn đoán `yUTu8PhWBuZc` đã chạy trên Colab, trả về Linux, Python 3.13.18, 2 CPU, RAM 12,67 GiB, `COLAB_RELEASE_TAG=release-colab-external-images_20261029-060149_RC00`, có `/content`, không phát hiện NVIDIA GPU. Các dấu hiệu xác nhận hosted Colab hiện dùng CPU. Không chạy huấn luyện trong lần kiểm tra này. Các ghi chú chưa ghép nối ở phần lịch sử bên dưới mô tả những lần thử trước, không phải trạng thái hiện tại.

Đã cài `uv` trong `.venv-colab-mcp` của dự án và thêm server `colab` vào cấu hình MCP của Codex bằng CLI đi kèm ứng dụng. Đây là cầu nối chạy nhẹ trên máy cá nhân; không dùng nó để huấn luyện local.

Nguồn chính thức: https://github.com/googlecolab/colab-mcp

Đã cố định phiên bản mã nguồn ở commit `b9ab3899e0f1fa493390b1fd6d54aa2e464ecdf1`. Server dùng Python 3.13 do uv quản lý.

Đã cập nhật launcher `colab_bridge_launcher.py` để server chỉ bind `127.0.0.1`. Trong log macOS trước đó, `localhost:0` tạo hai cổng khác nhau cho IPv4/IPv6; URL ghép nối có thể chọn cổng khác với địa chỉ browser truy cập. Launcher giữ nguyên kiểm tra origin/token của server Google. Đã kiểm tra lại MCP initialize/list_tools thành công và chỉ có một listener IPv4. Chưa xác nhận ghép nối notebook thực tế. Cần mở lại ứng dụng để nạp launcher mới; URL cũ thuộc tiến trình đã đóng không thể dùng tiếp.

Launcher còn bảo vệ tác vụ khởi tạo proxy bằng `asyncio.shield`. Mã upstream dùng `wait_for(gather(..., self._start_task))`, khiến hết thời gian chờ ở tool cũng hủy tác vụ khởi tạo lâu dài. Khi người dùng bấm Connect sau đó, WebSocket có thể mở rồi đóng mà không hoàn tất MCP handshake. Bản sửa đã được kiểm tra bằng `test_colab_bridge_retry.py`: chờ hết hạn trước, sau đó frontend mô phỏng xác thực và hoàn tất initialize/tools/list, rồi đóng bình thường. Đây là mô phỏng giao thức cục bộ, không phải xác nhận đã kết nối Google Colab thực tế.

Sau lần ghép nối trả về `true`, catalog công cụ của chat vẫn chỉ có bootstrap. Đã thêm `list_colab_tools` và `call_colab_tool` ngay khi khởi động để không phụ thuộc vào cập nhật tool động. Chúng chỉ chuyển tiếp qua transport Colab đã ghép nối và chỉ gọi tên tool mà Colab thực sự công bố. Đã kiểm tra MCP handshake có đủ cả ba công cụ; mô phỏng kiểm tra chuyển tiếp list/call thành công và từ chối tên tool không được công bố. Cần nạp lại tiến trình MCP để dùng bản này. Chưa có kết quả kiểm tra runtime Colab thật.

## Tiếp tục kết nối

1. Đóng và mở lại ứng dụng chat để nạp cấu hình MCP mới.
2. Quay lại chat dự án này, nhắn: “Kết nối MCP Colab và kiểm tra runtime đám mây”.
3. Khi công cụ `open_colab_browser_connection` xuất hiện, dùng nó để mở phiên ghép nối Colab. Hoàn tất đăng nhập/xác nhận ghép nối nếu Colab yêu cầu. Không gửi mật khẩu hoặc mã xác thực vào chat.
4. Xác nhận kết nối thành công, danh sách công cụ notebook xuất hiện và runtime thực sự ở Colab trước khi chạy mô hình. Không dùng local runtime.

Việc thêm cấu hình không đồng nghĩa đã ghép nối Google hoặc đã chạy notebook. Client cần hỗ trợ thông báo thay đổi danh sách công cụ (`notifications/tools/list_changed`) theo yêu cầu của server Google. Nếu không thấy công cụ sau khi mở lại ứng dụng, cần kiểm tra tính tương thích và log MCP; không chuyển lại sang huấn luyện local.

## Kiểm tra cấu hình

```bash
codex mcp get colab
```

Gỡ kết nối khi không cần:

```bash
codex mcp remove colab
```

Notebook đã chuẩn bị cho hosted Colab: `DS102_Colab_v3.ipynb`. Chuyển nơi chạy không tự cải thiện điểm số; hướng thử nghiệm mô hình tiếp theo sẽ chạy trên Colab sau khi kết nối được xác nhận.
