# Thiết kế thí nghiệm

## Câu hỏi nghiên cứu

1. Phương pháp đề xuất có tốt hơn MOMA-TD3 của paper và hơn các heuristic không cần học không?
2. Lợi thế đến từ prior, từ γ, hay từ cả hai? Câu này được trả lời bằng ablation 2×2.
3. Policy học được hành vi gì, và vì sao hành vi đó tốt hơn?

## Thiết kế 2×2

|  | γ 0.99 | γ 0.9 |
|---|---|---|
| Không prior | `ma3` (paper) | `ma3_g90` |
| Có prior | `ma3_prior_g99` | `ma3_prior` (đề xuất) |

Mỗi ô gồm 5 seed train (0–4), mỗi seed train 1 triệu bước. Các cấu hình giống hệt nhau, chỉ khác γ, prior và `save_dir`. Các file cấu hình nằm trong `config/train/moma_ac_ma3*.yaml`.

## Tách seed để tránh "học tủ"

| Dải seed | Dùng cho |
|---|---|
| Seed của episode lúc train | Huấn luyện |
| 100000+ | Validation: đã dùng khi phát triển `ma3_prior` (chọn γ, bán kính tam giác) |
| **200000+** | **Test:** chỉ dùng một lần, sau khi đã chốt phương pháp, cho mọi phương pháp kể cả heuristic |

Cách tách này làm theo khuyến nghị của Eimer, Lindauer & Raileanu (ICML 2023, Sec. 6): dùng seed tuning và seed test riêng, và ít nhất 5 seed.

## Cách đánh giá

- **Số episode và trọng số:** 20 episode test (seed 200000–200019) × 30 trọng số ω. Ba trọng số đầu là 3 góc (1,0,0), (0,1,0), (0,0,1); 27 trọng số còn lại rút từ Dirichlet với seed cố định.
- **Common random numbers:** mọi phương pháp chạy trên cùng các episode, nên so sánh được theo từng cặp.
- **Chạy trên CPU** (`eval_checkpoint.py`). Vòng lặp kín rất nhạy: chỉ cần lệch hành động 1e-6 là QoS ở một góc có thể đổi ±0.03, nên CPU, GPU và Kaggle cho số hơi khác nhau. Eval lúc train (3 episode) chỉ dùng để vẽ đường học, không dùng để kết luận.

## Chỉ số

- **HV (hypervolume):** thể tích mà front Pareto chiếm so với điểm tham chiếu `HV_REF_POINT`. Đây là chỉ số chính; càng lớn càng tốt.
- **EUM (expected utility):** trung bình `ω · reward` của policy tại chính trọng số ω đó, lấy trên 30 trọng số.
- **eu_front:** ở mỗi ω, lấy điểm tốt nhất trên front.
- **Tỉ lệ UE đạt đủ SLA theo slice:** chỉ để báo cáo, không nằm trong reward.
- **Hành vi** (`eval_trajectories.py`): tốc độ, khoảng cách tới đỉnh tam giác, độ phủ, công suất, phần băng thông. Đo ở 3 góc, bỏ 3 bước đầu là giai đoạn bay ra.

## Thống kê (`aggregate_seeds.py`)

- **Điểm của một phương pháp:** trung bình qua 5 seed. Front của mỗi seed là trung bình 20 episode.
- **Khoảng tin cậy 95%:** bootstrap hai tầng, 2000 lần. Mỗi lần lấy lại seed rồi lấy lại episode, nên khoảng tin cậy phản ánh cả khác biệt giữa các lần train lẫn nhiễu của eval. Ý tưởng này giống stratified bootstrap của Agarwal et al. (NeurIPS 2021).
- **Chênh lệch giữa hai phương pháp:** các episode được lấy lại với cùng chỉ số cho mọi phương pháp, nên chênh lệch là so cặp trên cùng episode.
- **P(improvement):** xác suất một seed của phương pháp A có HV cao hơn một seed của B (Agarwal et al. 2021, Sec. 5). Giá trị 1.0 nghĩa là mọi seed của A đều thắng mọi seed của B.

## Điều kiện công bằng giữa các phương pháp

- Cùng môi trường `v1.1`, cùng ngân sách 1 triệu bước, và cùng siêu tham số (trừ γ và prior).
- Heuristic xuất phát từ vị trí ngẫu nhiên giống các agent và phải bay tới tam giác. Riêng "tam giác đứng yên" được đặt sẵn tại chỗ, không tốn chi phí bay ra.
- γ 0.9 không được tune với ngân sách như nhau cho mọi phương pháp, nhưng ablation cho `ma3` thêm γ 0.9 (`ma3_g90`), nên baseline vẫn được hưởng cùng giá trị γ.
