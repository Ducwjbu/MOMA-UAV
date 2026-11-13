# Ghi chú cho báo cáo môn Dự án khoa học

Thư mục này gom mọi thứ cần để viết báo cáo: kết quả, cách làm thí nghiệm, cách giải thích, hạn chế và tài liệu tham khảo. Số liệu chốt ngày 2026-11-10, trên môi trường tag `v1.1`. Hạn nộp khoảng 2026-12-07 đến 12-14.

| File | Nội dung |
|---|---|
| [01_bai_toan_va_phuong_phap.md](01_bai_toan_va_phuong_phap.md) | Bài toán 3 UAV, môi trường, các phương pháp so sánh, phương pháp đề xuất |
| [02_thiet_ke_thi_nghiem.md](02_thiet_ke_thi_nghiem.md) | Seed, số bước, cách eval, thống kê, cấu hình |
| [03_ket_qua.md](03_ket_qua.md) | Toàn bộ bảng số liệu và các hình |
| [04_giai_thich_de_hieu.md](04_giai_thich_de_hieu.md) | Giải thích kết quả bằng lời thường, dùng khi trình bày |
| [05_han_che.md](05_han_che.md) | Hạn chế và hướng cho khoá luận |
| [06_tai_lieu_tham_khao.md](06_tai_lieu_tham_khao.md) | Tài liệu tham khảo kèm phần cụ thể đã dùng |

## Dàn ý đề xuất cho báo cáo

1. **Mở đầu:** UAV làm trạm phát sóng tạm thời, network slicing (eMBB, URLLC, mMTC), ba mục tiêu xung đột nhau (QoS, năng lượng, công bằng).
2. **Bài toán và môi trường:** xem file 01.
3. **Phương pháp:** MOMA-TD3 của paper làm baseline, họ heuristic, phương pháp đề xuất (residual policy trên đội hình tam giác, γ 0.9).
4. **Thiết kế thí nghiệm:** xem file 02.
5. **Kết quả:** HV, ablation 2×2, front Pareto, hành vi (quỹ đạo), SLA. Xem file 03.
6. **Thảo luận và hạn chế:** xem file 04 và 05.
7. **Kết luận và hướng phát triển:** cho khoá luận.

## Tạo lại số liệu và hình

Mọi lệnh chạy từ thư mục gốc repo. Checkpoint không có trong git; bản giải nén từ Kaggle nằm ở `runs/` trên máy.

```bash
# Hình (PNG + PDF) và bảng hành vi vào figures/
py plot_results.py --results results-v1.1 --out figures

# Gộp các seed (đã có sẵn trong results-v1.1/eval_test/agg.csv)
py aggregate_seeds.py --method ma3 "results-v1.1/eval_test/ma3_s*_actor.npy" \
    --method ma3_g90 "results-v1.1/eval_test/ma3_g90_s*_actor.npy" \
    --method ma3_prior_g99 "results-v1.1/eval_test/ma3_prior_g99_s*_actor.npy" \
    --method ma3_prior "results-v1.1/eval_test/ma3_prior_s*_actor.npy" \
    --family heuristics results-v1.1/eval_test/heuristics.npy \
    --family triangle results-v1.1/eval_test/triangle_formation.npy \
    --baseline ma3 --baseline heuristics --baseline ma3_g90 --baseline ma3_prior_g99 \
    --out results-v1.1/eval_test/agg

# Cần checkpoint: eval một checkpoint trên episode test, và ghi quỹ đạo
py eval_checkpoint.py --config config/train/moma_ac_ma3_prior.yaml \
    --checkpoint runs/saved_models_moma_ac_ma3_prior/seed_0/checkpoints/checkpoint_step_1000000.pth \
    --seed_base 200000 --out evals/ma3_prior_s0
py eval_trajectories.py --out results-v1.1/eval_test/trajectories
```

Commit chứa toàn bộ kết quả: `9e9e289` (số liệu), `4681c7b` (hình, quỹ đạo, chẩn đoán eMBB).

## Script chẩn đoán (`scripts/`)

Mỗi câu nhận định trong báo cáo, `CLAUDE.md` và `docs/thesis_brainstorm.md` mà không đến từ `eval_checkpoint.py` đều có một script ở đây. Chạy từ gốc repo; docstring đầu mỗi file ghi câu nó kiểm và lệnh chạy. Kết quả ghi vào `evals/diag/` (không commit). Các script chẩn đoán dùng seed validation 100000+; riêng `controller_fine_grid.py` và các script eMBB kiểm lại số đã báo trên seed test.

| Script | Kiểm câu nào |
|---|---|
| `embb_diag.py`, `embb_bound.py` | SLA eMBB 1–2% là do kịch bản (mục 8 của `03_ket_qua.md`) |
| `position_signal.py` | Vị trí UAV giải thích bao nhiêu phương sai QoS của một bước (lý do cần prior) |
| `critic_bias.py` | Critic với γ 0.99 lệch khỏi giá trị thật bao nhiêu (lý do chọn γ 0.9) |
| `triangle_kmeans.py` | Tam giác r = 600 m so với vị trí phủ tối ưu theo 3-means |
| `hold_test.py` | Bay qua lại có giúp QoS không, và cái giá với SLA URLLC |
| `urllc_failures.py` | UE URLLC trượt SLA vì KPI nào |
| `queue_pooling.py` | Phần năng lực phục vụ của hàng đợi chung đến từ UE mMTC |
| `ue_population.py` | Số UE đến thực tế so với mức hiệu chỉnh 91 |
| `controller_fine_grid.py` | Bộ điều khiển ω so với ω cố định tốt nhất trên lưới 0.025 |
| `energy_terms.py` | Công suất phát chiếm bao nhiêu trong mục tiêu năng lượng |
