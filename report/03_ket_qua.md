# Kết quả (seed test 200000+, 20 episode, 5 seed)

Nguồn: `results-v1.1/eval_test/agg.csv`, `agg_seeds.csv`, `{method}_s{n}.csv`, `heuristics.csv`, `trajectories.csv`. Hình nằm trong `figures/` (tạo bằng `py plot_results.py`).

## 1. Hypervolume (hình `fig_hv`)

| Phương pháp | HV [CI 95%] | Độ lệch giữa seed | EUM [CI 95%] | eu_front |
|---|---|---|---|---|
| **`ma3_prior` (đề xuất)** | **0.332** [0.324, 0.340] | **0.001** | **0.309** [0.305, 0.314] | 0.313 |
| `ma3_prior_g99` | 0.290 [0.265, 0.309] | 0.028 | 0.288 [0.276, 0.297] | 0.291 |
| Họ heuristic | 0.274 [0.265, 0.284] | — | — | 0.265 |
| `ma3_g90` | 0.268 [0.252, 0.285] | 0.016 | 0.278 [0.267, 0.288] | 0.281 |
| `ma3` (paper) | 0.176 [0.100, 0.239] | 0.092 | 0.185 [0.069, 0.261] | 0.195 |
| Tam giác đứng yên | 0.157 [0.145, 0.172] | — | — | 0.173 |

HV của từng seed (seed 0 → 4):

| Phương pháp | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
|---|---|---|---|---|---|
| `ma3` | 0.251 | 0.168 | 0.247 | **0.026** (sụp) | 0.188 |
| `ma3_g90` | 0.290 | 0.277 | 0.253 | 0.265 | 0.253 |
| `ma3_prior_g99` | 0.240 | 0.304 | 0.297 | 0.306 | 0.304 |
| `ma3_prior` | 0.330 | 0.331 | 0.333 | 0.333 | 0.331 |

## 2. Chênh lệch HV từng cặp

| So sánh | ΔHV [CI 95%] | P(improvement) |
|---|---|---|
| Đề xuất vs `ma3` (paper) | **+0.156** [+0.093, +0.233] | 1.0 |
| Đề xuất vs họ heuristic | **+0.058** [+0.048, +0.068] | 1.0 |
| Đề xuất vs `ma3_g90` | +0.064 [+0.048, +0.081] | 1.0 |
| Đề xuất vs `ma3_prior_g99` | +0.041 [+0.025, +0.067] | 1.0 |
| `ma3_g90` vs `ma3` | +0.092 [+0.027, +0.174] | 1.0 |
| `ma3_prior_g99` vs `ma3` | +0.114 [+0.045, +0.196] | 0.92 |
| `ma3_g90` vs heuristic | −0.006 [−0.023, +0.012] | 0.4 |
| `ma3_prior_g99` vs heuristic | +0.016 [−0.009, +0.036] | 0.8 |
| `ma3` vs heuristic | −0.098 [−0.175, −0.035] | 0.0 |

## 3. Ablation 2×2 (hình `fig_ablation`)

|  | γ 0.99 | γ 0.9 |
|---|---|---|
| Không prior | 0.176 | 0.268 (+0.092) |
| Có prior | 0.290 (+0.114) | **0.332 (+0.156)** |

- Mỗi thành phần đứng riêng chỉ đưa HV lên ngang họ heuristic: khoảng tin cậy của chênh lệch so với heuristic đều chứa 0.
- Chỉ khi có cả prior lẫn γ 0.9 thì mới vượt heuristic rõ ràng.
- Hai tác dụng không cộng dồn: riêng lẻ là +0.092 và +0.114, tổng 0.206, nhưng kết hợp chỉ được +0.156. Nhiều khả năng cả hai cùng sửa một phần của cùng một vấn đề: critic bi quan khiến agent không học được cách dàn đội hình.
- **Đính chính so với pilot:** ở mốc 100k bước, prior với γ 0.99 không hơn `ma3`; ở 1M bước thì hơn.

## 4. Đường học (hình `fig_learning_curves`)

Eval lúc train, 3 episode, chỉ dùng để xem xu hướng. HV ở 1M bước, trung bình 5 seed:

| `ma3` | `ma3_g90` | `ma3_prior_g99` | `ma3_prior` |
|---|---|---|---|
| 0.189 | 0.282 | 0.311 | 0.361 |

Phương pháp đề xuất dẫn đầu từ những bước đầu, và vùng min–max qua 5 seed rất hẹp. `ma3` có vùng min–max rất rộng vì seed 3 không học được.

## 5. Ba góc trọng số (QoS / năng lượng / công bằng của reward)

| Góc | `ma3` | `ma3_g90` | `ma3_prior_g99` | `ma3_prior` | Tam giác đứng yên |
|---|---|---|---|---|---|
| ω = QoS | 0.220 / −0.105 / 0.335 | 0.328 / −0.072 / 0.463 | 0.354 / −0.106 / 0.464 | **0.403** / −0.108 / **0.514** | 0.277 / −0.304 / 0.425 |
| ω = năng lượng | 0.092 / −0.087 / 0.163 | **0.158** / −0.023 / **0.272** | 0.100 / −0.028 / 0.169 | 0.126 / **−0.021** / 0.221 | |
| ω = công bằng | 0.203 / −0.112 / 0.341 | 0.314 / −0.072 / 0.473 | 0.345 / −0.104 / 0.496 | **0.374** / −0.110 / **0.517** | |

- **Heuristic tốt nhất về QoS** (đứng yên ở tam giác, ưu tiên băng thông cho URLLC): 0.362 / −0.303 / 0.504. Phương pháp đề xuất ở góc QoS đạt QoS và công bằng cao hơn, với năng lượng chỉ khoảng một phần ba, tức là trội hơn hẳn (dominate).
- **Heuristic dao động** quanh tam giác (bay 11.4 m/s, ưu tiên URLLC): QoS 0.268, năng lượng −0.118. Ở cùng tốc độ bay, policy học được đạt QoS 0.403.
- **Trung thực:** ở góc năng lượng, `ma3_g90` có QoS và công bằng cao hơn phương pháp đề xuất với mức năng lượng gần như nhau.

Bảng đầy đủ 10 heuristic nằm trong `results-v1.1/eval_test/heuristics.csv`.

## 6. Front Pareto (hình `fig_fronts`)

Mỗi điểm là policy tại một trong 30 trọng số ω, trung bình 20 episode × 5 seed.
- Trừ điểm ở góc năng lượng, các điểm của phương pháp đề xuất nằm ở vùng QoS 0.34–0.40 với năng lượng −0.05 đến −0.11, tức là trên và bên phải các điểm của `ma3` và của heuristic.
- `ma3` co cụm quanh QoS khoảng 0.21, gần như không thay đổi theo ω.

## 7. Hành vi (hình `fig_trajectories`, `figures/behaviour.csv`)

Trung bình 5 seed × 20 episode, từ bước 3 trở đi:

| Phương pháp | Góc | Tốc độ (m/s) | Cách đỉnh tam giác (m) | Công suất | Độ phủ |
|---|---|---|---|---|---|
| `ma3` | QoS | 11.3 | 852 | 0.94 | 0.41 |
| `ma3_g90` | QoS | 12.9 | 711 | 1.00 | 0.51 |
| `ma3_prior_g99` | QoS | 11.2 | 375 | 1.00 | 0.52 |
| `ma3_prior` | QoS | 11.6 | 363 | 1.00 | 0.53 |
| `ma3` | năng lượng | 11.6 | 1046 | 0.42 | 0.26 |
| `ma3_g90` | năng lượng | 14.9 | 855 | 0.54 | 0.34 |
| `ma3_prior_g99` | năng lượng | 14.3 | 679 | 0.36 | 0.27 |
| `ma3_prior` | năng lượng | 14.7 | 604 | 0.42 | 0.30 |

- **Không policy nào hover.** Ở góc QoS, mỗi UAV của phương pháp đề xuất bay qua lại trên một đoạn thẳng gần đỉnh tam giác của nó. Hành vi này giống nhau ở cả 5 seed: tốc độ 10.9–12.2 m/s, khoảng cách tới đỉnh 353–375 m.
- **Lý do:** theo mô hình Zeng, bay chậm tốn ít năng lượng hơn lơ lửng, nên vừa giữ được vùng phủ vừa tiết kiệm pin.
- **Góc năng lượng:** mọi phương pháp đều bay đường dài gần tốc độ tối đa (17 m/s), với công suất khoảng 0.4.
- **Seed 3 của `ma3`** (HV 0.026) là policy duy nhất gần như đứng yên (71–85% số bước), và đứng ở mép khu vực, cách các đỉnh 1.3–1.6 km.
- **Băng thông:** mọi policy học được đều dành phần lớn nhất cho URLLC. Ở góc QoS, phương pháp đề xuất giảm phần cho mMTC.

## 8. Tỉ lệ UE đạt đủ SLA

| Slice | `ma3` | `ma3_g90` | `ma3_prior_g99` | `ma3_prior` | Tam giác đứng yên |
|---|---|---|---|---|---|
| URLLC, trung bình 30 ω | 0.118 | 0.142 | 0.292 | **0.341** | 0.246 |
| URLLC, ω = QoS | 0.147 | 0.189 | 0.371 | **0.400** | 0.246 |
| mMTC, trung bình 30 ω | 0.248 | 0.338 | 0.338 | **0.360** | 0.282 |
| eMBB, trung bình 30 ω | 0.014 | 0.018 | 0.017 | 0.019 | 0.017 |

**Chẩn đoán eMBB** (`report/scripts/embb_diag.py` và `embb_bound.py`, chạy từ gốc repo, 5 episode test, góc QoS):

| Chỉ số | Đề xuất | Heuristic tốt nhất |
|---|---|---|
| UE eMBB có tải được phủ sóng | 27% | 43% |
| Thông lượng trung vị của UE eMBB được phủ | ~4 Mbps | ~3 Mbps |

- Trong các UE eMBB được phủ mà không đạt SLA, 99–100% hỏng vì thông lượng.
- 56% UE eMBB ở trong nhà, và chỉ 12–20% trong số này được phủ.
- **Cận trên:** giữ tam giác và dồn toàn bộ băng thông cho eMBB thì tỉ lệ đạt SLA eMBB cũng chỉ 4.9%, URLLC và mMTC về 0.

Kết luận: mức 1–2% đến từ kịch bản, không phải lỗi chỉ số. Xem `05_han_che.md`.
