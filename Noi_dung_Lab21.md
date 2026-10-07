# Hiểu bài toán và lấy repo

**Một câu tóm tắt lab:** fine-tune một model mở bằng LoRA — rồi **chứng minh** nó thắng được chính model đó khi đã được prompt tử tế. Nếu không chứng minh được, phát hiện ra điều đó cũng được tính điểm đầy đủ.

Lab bắt bạn trả lời hai câu hỏi:

- **1. Phần được tính loss có đúng là câu trả lời không?** (NB1)
- **2. Bản fine-tune có thắng base model đã prompt tử tế không — và bạn có phát hiện được nếu nó không thắng?** (NB2 đóng băng mốc, NB5 phán quyết)

**Repo:** https://github.com/VinUni-AI20k/Day21-Track3-Finetuning-Lab

**Chạy trên Colab (khuyến nghị):** mở `colab/Lab21_RUN_ALL.ipynb` → Runtime → Change runtime type → **T4 GPU** → chạy lần lượt ô 1 → 4.

> 💡 Mỗi lần repo đổi, hãy **mở lại tab** (reload), đừng chỉ reconnect — Colab chỉ đọc notebook từ GitHub một lần, lúc mở URL.

**Thí nghiệm của bạn — bạn tự chọn:**

|  | Mặc định | Đổi thế nào |
|---|---|---|
| Base model | theo tier (Qwen3.5-4B trên T4) | `BASE_MODEL=<hf-id>` trong `.env` |
| Dataset | 250 ticket CSKH → JSON triage | xem README, mục "Đổi dataset của riêng bạn" |
| Report | mẫu `submission/REPORT.md` | tự viết theo cấu trúc của bạn |

Hai điều không đổi: **khai báo** lựa chọn và lý do trong report; mốc (NB2) **đóng băng trước khi train**, cùng một base model cho mốc lẫn bản fine-tune.

---

# NB1 — Dữ liệu, chat template và loss mask

**Mục tiêu:** chứng minh phần được tính loss **đúng bằng** câu trả lời. ~25 giây, không cần GPU.

- In chuỗi **sau `apply_chat_template`** và giải mã ngược các vị trí có `labels != -100`.
- Hai assert phải xanh: câu trả lời nằm trong loss, câu hỏi không nằm trong loss → `results/mask_proof.json`.
- `template_check.json`: template có giữ khối `<think>` không?
- Đặt `max_length` theo **p95 đo được** (`token_stats.json`), không đoán.

**Đổi base model?** Chạy `python scripts/check_mask_agreement.py` trước khi train. Script so mask của lab với hai đường của TRL: đường tokenizer (template thiếu `{% generation %}` ⇒ mask rỗng, chỉ cảnh báo) và đường `SFTTrainer` (vá template hoặc báo lỗi). Mask phải được chứng minh lại cho từng model.

> 💡 Mất trắng điểm 1.1 nếu `supervised_fraction ≥ 0.95` — nghĩa là bạn đang tính loss cả trên prompt.

---

# NB2 — Đóng băng mốc: ba baseline trước khi train

**Mục tiêu:** đo baseline trước khi train, rồi đóng băng. ~17–23 phút trên T4.

- (a) base model + prompt đơn giản
- (b) base model + prompt **đã tối ưu** + few-shot — đây là đối thủ thật
- (c) sau này: bản fine-tune

Kết quả: `results/baselines_frozen.json`. Yêu cầu `(b) > (a)`.

> 💡 `make verify` kiểm tra checksum tập eval và SHA của prompt (b). Làm mạnh (b): hoan nghênh. **Làm yếu (b) để fine-tune trông thắng là gian lận.**

---

# NB3 — Train cấu hình đúng

**Mục tiêu:** train cấu hình "vùng không hối tiếc" của deck (§11). ~15–25 phút trên T4.

| Nút vặn | Giá trị | Deck |
|---|---|---|
| `target_modules` | toàn bộ linear của text decoder | §11.2 |
| `learning_rate` | ≈10× LR full-FT | §11.3 |
| batch hiệu dụng | < 32 | §11.4 |
| `alpha` | `2r` | §10.3 |

NB3 train trên **đúng mask NB1 đã chứng minh** (pre-tokenize), không dựa vào cờ `assistant_only_loss`. In `layer_types` của chính model bạn dùng.

Kết quả: `adapters/correct/` và dòng `correct` trong `results/runs.csv` (loss + VRAM).

---

# NB4 — Mổ xẻ ba cấu hình sai

**Mục tiêu:** ba run đối chứng, **cùng số step** với `correct`, mỗi run đổi đúng **một** biến. ~45–60 phút trên T4.

| Run | Đổi gì |
|---|---|
| `attn_only` | chỉ gắn LoRA vào attention — rank **đã khớp ngân sách tham số** (`matched_rank()`), sai lệch < 5% |
| `wrong_lr` | LR thang full-FT (1×) |
| `qlora` | 4-bit QLoRA |

> 💡 So `q,v @ r=16` với `all-linear @ r=16` là **so ngân sách**, không phải so vị trí. Đây là mục dễ mất điểm nhất.

Nếu NB4 đứt giữa chừng: adapter đã lưu được bỏ qua; `ONLY=qlora` để train lại đúng một run.

---

# NB5 — Đánh giá bốn nhóm và phán quyết

**Mục tiêu:** chấm bản fine-tune so với mốc đã đóng băng. ~21 phút trên T4.

| Nhóm | Đo bằng |
|---|---|
| target | độ chính xác từng trường so với nhãn |
| regression | câu hỏi phổ thông — fine-tune **không được** làm hỏng |
| format | JSON parse được + đủ khoá |
| latency | ms/mẫu, greedy decode |

- Xếp hạng bốn run bằng **điểm target ở NB5**, không bằng `final_loss` của NB4.
- `results/verdict.json`: PASS hay FAIL **đều được điểm** — miễn bạn diễn giải nó.
- ≥5 ví dụ định tính, trong đó **≥2 ca fine-tune THUA**.

**NB6 (tuỳ chọn, +3):** merge adapter, assert điểm không tụt, hot-swap ≥2 adapter.

**Bị bó thời gian?** `EVAL_LIMIT=8` hoặc `EPOCHS=1` — nhưng **nộp bài thì để mặc định**; `results/` ghi lại nếu bạn chạy chế độ rút gọn.

---

# Report và nộp bài

**Trước khi nộp:** chạy `make verify` (ô 4 trên Colab). Gatekeeper kiểm tra mask proof, ngân sách `attn_only`, checksum tập eval, SHA prompt (b) và `(b) > (a)`.

**Report do bạn tự cấu trúc** (mẫu `submission/REPORT.md` chỉ là gợi ý), nhưng phải có:

1. Model + dataset đã chọn, và **lý do**
2. Bằng chứng mask (NB1)
3. Mốc đã đóng băng (NB2)
4. Kết quả so sánh bốn nhóm + bốn run
5. Phán quyết và diễn giải
6. Điều bạn học được — cụ thể, không chung chung

Mọi con số trong report phải **khớp** với file trong `results/` — grader kiểm tra chéo.

**Nộp** (chọn một): ZIP gọn (`submission/REPORT.md` + `results/` + `adapters/correct/` + notebooks) · GitHub + HuggingFace Hub (+2 điểm) · code-only. Cả ba đều **bắt buộc có `results/` đầy đủ**.

**Thang điểm:** pipeline 30 · thí nghiệm công bằng 25 · đánh giá & phán quyết 25 · report 20 · thưởng tối đa +15. Chi tiết: `rubric.md` trong repo.

> 💡 Điểm không nằm ở chỗ fine-tune của bạn thắng. Điểm nằm ở chỗ **bạn biết nó có thắng hay không**.
