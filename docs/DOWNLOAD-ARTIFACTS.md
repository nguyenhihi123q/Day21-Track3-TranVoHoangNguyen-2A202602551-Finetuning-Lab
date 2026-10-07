# Khôi phục adapter và ZIP bài nộp

Repo public fork này không cho phép tải object Git LFS mới. Các file trọng số và ZIP được lưu thành phần `.part001`, `.part002`… tối đa 64 MiB mỗi phần.

Sau khi clone repo, chạy tại thư mục gốc:

```bash
python scripts/restore_large_files.py
```

Script ghép lại bốn adapter và hai ZIP ở đường dẫn ban đầu, đồng thời kiểm tra SHA-256 theo `large_files_manifest.json`. Nội dung file được giữ nguyên byte. Cần khoảng 650 MB dung lượng trống để khôi phục.

Để nộp bài, dùng `submission/packages/lab21_2A202602551_OptionA.zip` sau khi ghép.
