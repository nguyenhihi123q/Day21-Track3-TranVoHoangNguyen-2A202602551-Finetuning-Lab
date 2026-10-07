# Phiên Colab của bài Lab 21

Notebook đã tải lên tài khoản Google của bạn:
https://colab.research.google.com/drive/17IRjOtA8na6IxVH345m__5dTOwVz46ks

Bản local: `colab/Lab21_COLAB_FULL.ipynb`.
Tạo lại từ các file được Git theo dõi: `python scripts/prepare_colab_full.py`.
Notebook chứa snapshot mã lab để tránh lỗi tải từ GitHub; không đóng gói `.env` hay token.

## Cấu hình

- GPU: T4; đã thấy Colab cấp Tesla T4 với 14.56 GB VRAM.
- Tier: T4, base model theo `src/labkit/config.py`.
- Epochs: 2, loss mask: assistant-only, toàn bộ tập đánh giá (không đặt EVAL_LIMIT).
- NB1 → NB2 → NB3 → NB4 → NB5, sau đó NB6 để lấy bằng chứng thưởng.
- Mã training/evaluation gốc không bị sửa; report hiện vẫn là mẫu cần hoàn thiện.

## Theo dõi và khôi phục

1. Ô Setup cài dependency. Nếu Colab yêu cầu restart, restart rồi chạy lại Setup.
2. Ô kiểm tra môi trường phải thành công trước khi chạy NB1.
3. Chỉ chạy NB3 sau khi NB2 đã ghi baseline đóng băng.
4. Sau mỗi notebook, ZIP checkpoint được tạo trong `/content`. Mở Files bên trái,
   chọn menu của ZIP → Download để sao lưu xuống máy. AUTO_DOWNLOAD mặc định tắt
   vì tải tự động có thể làm trình duyệt trong ứng dụng mất kết nối.
5. NB4 tự bỏ qua adapter đã lưu trong cùng runtime. Nếu runtime bị xoá, phải khôi phục
   ZIP checkpoint vào `/content/Day21-Finetuning-Lab` trước khi chạy tiếp.
6. Khi một ô lỗi, dừng ở ô đó; xem log và sửa lỗi trước khi chạy tiếp.
   Không chạy lại NB2 hoặc đổi eval để làm baseline yếu đi sau khi đã train.
7. ZIP chứa `results/`, log, adapter (trừ merged), dữ liệu và mã notebook.
   Sau NB6/final, tải `lab21_final_checkpoint.zip` để đem kết quả về workspace.

## Trước khi nộp

- Đọc `token_stats.json`: ghi p95, suggested_max_length và max_length thực tế của tier;
  nếu giữ giá trị tier khác gợi ý p95, giải thích lựa chọn trong report.
- Ghép bảng training `runs.csv` với điểm tác vụ `autopsy.json`; xếp hạng bằng target.
- Report dùng số liệu từ phiên này, không chép số tham khảo trong docs hoặc SIMULATION-FINDINGS.
- Ít nhất 5 ví dụ định tính và ít nhất 2 ca FT thua; đối chiếu nhãn và baseline (b).
- Diễn giải verdict, kết luận ít nhất 150 từ và phản tư cá nhân cụ thể.
- NB6 cần cả `merge_check.json` lẫn log hot-swap ít nhất hai adapter.
- Chạy `python scripts/verify.py` khi report đã hoàn thiện. FAIL do report mẫu là dự kiến
  trước bước viết report, nhưng phải giải quyết mọi lỗi khác trước khi nộp.

## Trạng thái đã quan sát

Setup hoàn tất và smoke check trả về exit 0. NB1 hoàn tất trong khoảng 23 giây:
`answer_is_supervised=true`, `question_is_masked=true`, supervised fraction `0.4149`
(39/94 token), p95 `98`, max `101`, suggested_max_length `256`, train/val `225/25`.
Đây là số quan sát trên output Colab, chưa tải JSON gốc về workspace.

Sau NB1, trình duyệt trong ứng dụng báo lỗi lưu tự động và kết nối “Resuming execution”
khi tải ZIP. Kết nối sau đó phục hồi; đã tắt AUTO_DOWNLOAD và cập nhật helper trên Drive.
Một thao tác tiếp tục hàng đợi đã chọn ô NB4 ngoài ý muốn. Run đó đã ngắt và các tiến trình
được dừng; xác nhận `baselines_frozen.json` chưa có, `runs.csv` chưa có và chưa có adapter.
Helper mới chặn train khi chưa có baseline, chặn NB4 khi chưa có adapter correct,
và dọn cả nhóm tiến trình khi bị ngắt. Nếu kết nối gặp lỗi, mở cùng notebook trên Chrome/Edge;
không tạo thêm phiên GPU. Chạy bằng nút của từng ô để xác nhận đúng bước đang chạy.

Kết nối và lưu đã phục hồi; Colab hiện hiển thị “All changes saved”. NB2 đã bắt đầu
chạy full eval (`target=50`, `regression=15`); đã quan sát baseline (a) target sinh batch 1/13,
2/13 trên Tesla T4. Đã bấm xếp các ô NB3, NB4, NB5, NB6 và kiểm tra cuối phía sau.
Chưa có điểm baseline hoặc kết quả train tại thời điểm ghi trạng thái này.

Chưa có kết quả train thật; không suy ra điểm từ việc Setup/NB1 chạy thành công.
