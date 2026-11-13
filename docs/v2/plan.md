# Kế hoạch môi trường v2 và kiến trúc bản 3 (dạng TIP)

Bắt đầu 2026-11-13. Nguồn yêu cầu: trang Notion "Khoá luận MOMA-UAV: hạn chế và câu hỏi cho thầy" (mục 1, 3, 3.1, 5) và `docs/thesis_brainstorm.md` (kiến trúc bản 3). Mỗi TIP xong thì nộp một Completion Report ở cuối file này (STATUS, FILES CHANGED, TEST RESULTS, ISSUES, DEVIATIONS, SUGGESTIONS).

## Ràng buộc chung (áp cho mọi TIP)

- **v1.1 không đổi.** Báo cáo môn học chạy trên `v1.1` đến giữa tháng 12. Mọi tính năng v2 là khoá config, mặc định tắt. Với `ma3.yaml` và `default.yaml`, output env phải giống `v1.1` từng bit, và smoke train cho trọng số giống từng bit. Kiểm sau mỗi TIP bằng `docs/v2/check_v11.py` (TIP 0).
- **Config v2 riêng:** `config/environment/v2.yaml`, `config/train/moma_ac_v2_*.yaml`.
- **Không đụng seed test 200000+** cho tới khi phương pháp v2 chốt; dùng 100000+ để phát triển.
- **Nguồn:** mọi tham số mới có trích dẫn đầy đủ (tác giả, năm, nơi đăng, Eq./Table) trong comment config và trong `docs/v2/sources.md`.
- Commit và push lên `duc` sau mỗi TIP đã kiểm.

## Các lựa chọn đã chốt khi chưa có ý thầy (đổi được bằng config)

| Điểm | Chọn | Nếu thầy chọn khác |
|---|---|---|
| Câu chuyện triển khai (Q1) | (b) khẩn cấp: mạng mặt đất hỏng, UAV phục vụ tạm | (a): đổi bảng slice trong config, thêm trạm macro (TIP để sau) |
| Mục tiêu (Q2) | Vector 4 chiều: SLA dịch vụ dữ liệu, SLA dịch vụ khẩn cấp (URLLC), SLA mMTC, năng lượng. Công bằng chỉ báo cáo | Giữ 3 mục tiêu cũ: khoá `reward.objectives` |
| Phạm vi (Q3) | Ca trực nhiều chuyến, đổi pin, 2–3 loại sự kiện | Episode 18 bước: tắt `shift` |
| Baseline tối ưu (Q4) | Có (TIP 9) | — |
| Lý thuyết (Q5) | Định lý đầy đủ cho tầng đội, mệnh đề + thí nghiệm cho tầng UAV (TIP 10) | — |

## Thay đổi sau kiểm độc lập (2026-11-13)

Một subagent không biết gì về dự án đã kiểm repo. Kết luận chính: không có lỗi làm sai kết quả chính (HV 0.332 tái tạo đúng từng episode). Những điểm đưa vào v2:

- **URLLC không có biên tin cậy.** PER dư 1e-5 đúng bằng mục tiêu, nên UE chỉ đạt SLA khi drop và mất gói do handover bằng 0 tuyệt đối. Vì vậy `sla_urllc` ở v1.1 chủ yếu đếm handover. TIP 1: ngân sách liên kết có biên, mất gói do handover tính riêng.
- **Hàng đợi chung gộp cả năng lực không có thật.** Năng lực của UE mMTC (khoảng 1e-6 Mbps) được cộng vào hàng đợi phục vụ eMBB. TIP 2: mỗi slice có hàng đợi riêng, phục vụ bằng chính năng lực của slice đó.
- **Mốc 18 bước bị coi là cắt ngang, không phải kết thúc, và quan sát không có thời gian.** TIP 4: quan sát có thời gian còn lại của ca và pin; kết thúc ca là terminal.
- **Bộ điều khiển chỉ "ngang ω cố định tốt nhất" khi lưới ω thô (bước 0.1).** Với lưới 0.025, ω cố định a = 0.25 tiết kiệm năng lượng hơn, nhưng có 31% episode vi phạm. TIP 8: oracle chọn trên lưới mịn, và theo cùng tiêu chí từng episode; thêm baseline luật ngưỡng (bang-bang) và RCPO.
- **Mục tiêu policy học khác ràng buộc được đo.** Policy tối đa hoá return chiết khấu γ 0.9, còn ràng buộc là trung bình không chiết khấu của episode. TIP 10 phải xử lý chỗ lệch này.
- **Phân phối ω khi train phủ kém vùng bộ điều khiển dùng.** Bộ điều khiển chạy trên các cạnh của đơn hình, mà Dirichlet(1,…,1) hiếm khi rút ở đó. TIP 8: train trộn thêm ω trên cạnh và mặt của đơn hình.
- **Công suất phát gần như miễn phí** (chỉ chiếm tối đa 0.017 của mục tiêu năng lượng), còn treo đắt hơn bay. Ghi rõ là đặc tính của mô hình Zeng; TIP 3 tính mạng ở bước con để chấm cả QoS trong lúc bay.

## Thứ tự và phụ thuộc

```
TIP 0 (kiểm v1.1) ─┬─ TIP 1 (SLA 5QI + mục tiêu theo slice) ── TIP 2 (hành động theo slice + scheduler)
                   │        └─ TIP 3 (bước con 10 s)
                   ├─ TIP 4 (ca trực, đổi pin) ── TIP 5 (sự kiện, tải theo giờ)
                   ├─ TIP 6 (UE có trọng số, hiệu chỉnh tải)   [sau TIP 1, 2]
                   └─ TIP 7 (prior theo nhu cầu)
TIP 8 (eval 4 mục tiêu, KPI theo thời gian, bộ điều khiển nhiều ràng buộc)  [sau TIP 1]
TIP 9 (baseline tối ưu)                                                       [sau TIP 2, 6]
TIP 10 (lý thuyết) chạy song song từ đầu
TIP 11 (pilot v2 trên Kaggle)                                                 [sau 1–8]
```

---

### TIP 0. Bộ kiểm v1.1 bit-for-bit

- **Task:** script `docs/v2/check_v11.py` ghi lại tham chiếu từ commit hiện tại: 3 episode × `ma3.yaml` và `default.yaml` với hành động ngẫu nhiên có seed (reward vector, info, vị trí UE và UAV mỗi bước), cộng trọng số sau smoke train. Lần sau so với tham chiếu.
- **Đạt khi:** chạy hai lần trên code hiện tại cho kết quả giống nhau; cố ý đổi một hằng số thì script báo lệch.

### TIP 1. Ngưỡng SLA theo 5QI và mục tiêu theo slice

- **Context:** `environment.py` (`_calculate_qos_reward`, `QoSProfile`, `info['sla_*']`), Notion mục 3.1.
- **Task:**
  - Bảng slice trong config: mỗi slice có tốc độ, ngân sách trễ (đã trừ phần mạng lõi), PER, tên 5QI. Câu chuyện (b): dữ liệu khẩn cấp 5QI 70 (200 ms − 10 ms lõi, 10⁻⁶ → xem ISSUES), thoại/điều khiển khẩn cấp 5QI 65/84, mMTC theo TR 38.913.
  - `reward.objectives: [sla_data, sla_critical, sla_mmtc, energy]`: reward mỗi slice là **surrogate trơn** của tỉ lệ UE đạt SLA. Mỗi UE có điểm `s_u = min_k σ(β·margin_k)`, trong đó margin là khoảng cách (theo log) tới ngưỡng của KPI k; slice lấy trung bình. β lớn thì gần chỉ số nhị phân. Chỉ số nhị phân vẫn báo cáo trong `info`.
  - Độ tin cậy tính theo đúng PER so với ngưỡng (không cho điểm theo "số 9" như v1.1), để policy thấy cái giá của handover với URLLC (phép thử đứng yên 2026-11-13).
  - Ngân sách lỗi có biên: PER dư của liên kết nhỏ hơn mục tiêu (ví dụ BLER 1e-5 với 2 lần truyền, dư 1e-10). Mất gói do handover và do drop cộng vào ngân sách đó. So sánh dùng dung sai tương đối thay vì so số thực bằng nhau.
- **Đạt khi:** với β → ∞ surrogate trùng `info['sla_*']` ở mọi bước (kiểm trên 5 episode); v1.1 không đổi (TIP 0); smoke train v2 4 mục tiêu chạy hết.

### TIP 2. Hành động theo slice và bộ lập lịch bên trong

- **Task:** khoá `action.mode: slice_share`: hành động mỗi UAV = di chuyển (3) + công suất (1) + tỉ lệ tài nguyên mỗi slice (3), tổng 7. Bên trong bước, bộ lập lịch proportional fair trong từng slice, RB thừa chuyển sang slice khác (dùng lại `rb_sharing: all`). Trainer lấy `act_dim` từ env thay vì từ `num_das_per_slice`.
- **Đạt khi:** tam giác đứng yên với tỉ lệ đều có SLA mỗi slice ≥ bản 13 chiều chia đều; không còn RB bỏ phí khi có UE có tải; v1.1 không đổi.

### TIP 3. Tính mạng ở bước con

- **Task:** khoá `substeps: 6` (10 s): UAV đi theo đoạn thẳng của bước; ở mỗi bước con tính LoS, association, SINR, rate, queue; KPI của bước là trung bình các bước con. Handover đếm ở bước con. Năng lượng không đổi (Zeng đã tính theo quãng bay).
- **Đạt khi:** với UAV đứng yên, kết quả trùng `substeps: 1`; thời gian chạy env được đo và ghi lại; trình xem hiện các bước con là khung thật.

### TIP 4. Ca trực và đổi pin

- **Task:** khoá `shift`: một episode là một ca (ví dụ 4 giờ = 240 bước). UAV quay về Home Point khi chạm năng lượng quay về (đã có), đổi pin mất `swap_steps` bước (nguồn: thời gian đổi TB60 của DJI M300), rồi bay ra lại. Dùng cơ chế `fleet` để đánh dấu UAV không hoạt động. Có `n_spare` UAV dự phòng.
- **Đạt khi:** trong một ca, mỗi UAV có ít nhất một lần đổi pin; số UAV hoạt động khớp log; năng lượng có hậu quả (thời gian không phục vụ).

### TIP 5. Tải theo giờ và sự kiện

- **Task:** hệ số tải theo giờ (nguồn: đo lưu lượng di động theo giờ, cần tìm); sự kiện: điểm nóng xuất hiện rồi tan (đám đông tụ ở điểm tập kết), UAV hỏng giữa ca, ngưỡng của nhà mạng đổi. Sự kiện rút từ RNG có seed, ghi trong `info['events']`.
- **Đạt khi:** cùng seed cho cùng chuỗi sự kiện bất kể policy; trình xem hiện sự kiện.

### TIP 6. UE có trọng số và hiệu chỉnh tải

- **Task:** mỗi UE mô phỏng đại diện `w` người dùng (tải và đếm SLA nhân `w`). Hiệu chỉnh tải danh nghĩa để policy tốt đạt các ngưỡng (Notion 3.1, điều kiện 1).
- **Đạt khi:** ở tải danh nghĩa, tam giác đứng yên với phân bổ hợp lý đạt khoảng 0.8–0.9 mỗi slice khẩn cấp (để ngưỡng 0.95 khó nhưng với tới được); bảng hiệu chỉnh ghi trong `sources.md`.

### TIP 7. Prior theo nhu cầu

- **Task:** đỉnh của prior = k-means có trọng số trên bản đồ mật độ (đã có `traffic_map`) thay cho tam giác cố định, gán bằng Hungarian (đã có). Khoá `prior_mode: demand | ngon`.
- **Đạt khi:** trên bản đồ có điểm nóng lệch tâm, prior theo nhu cầu cho coverage cao hơn tam giác khi đứng yên.

### TIP 8. Đánh giá 4 mục tiêu và bộ điều khiển nhiều ràng buộc

- **Task:** hypervolume tổng quát n chiều (thuật toán WFG, While et al. 2012, IEEE TEC 16(1)), kiểm bằng Monte Carlo; KPI theo thời gian (tỉ lệ thời gian vi phạm, thời gian hồi phục sau sự kiện); `eval_controller.py` nhận nhiều ràng buộc và hàng đợi ảo cục bộ cho ω_i; trình xem vẽ λ và ω theo thời gian.
- **Đạt khi:** HV n chiều trùng code 3D hiện có đến 1e-9 trên mọi file `pareto_points.csv` của `results-v1.1`; bộ điều khiển với ω cố định tái tạo `eval_checkpoint.py`.

### TIP 9. Baseline tối ưu

- **Task:** mỗi bước: đặt UAV bằng k-means có trọng số trên UE đang có tải, rồi chia tỉ lệ slice bằng bài toán tối ưu nhỏ (tối đa tổng SLA có trọng số với ràng buộc RB, giải bằng tìm kiếm trên lưới tỉ lệ dùng mô hình rate của env). Có quyền xem trạng thái thật (oracle), nên là cận trên thực dụng, không phải đối thủ ngang hàng.
- **Đạt khi:** chạy được trên v2, ghi cùng định dạng `.npy` với `eval_checkpoint.py`.

### TIP 10. Lý thuyết (song song)

- **Task:** `docs/v2/theory.md`: phát biểu và chứng minh đầy đủ, mỗi bước có trích dẫn; giải thích cho bạn từng bước. Dự kiến:
  1. Mệnh đề 1: hàm Lagrange của bài toán có ràng buộc bằng (hằng số dương) × tổng có trọng số với ω(λ) = (e₀ + λ)/(1 + ‖λ‖₁), trừ λᵀc. Hệ quả: policy có điều kiện theo ω tối ưu là best response của mọi λ.
  2. Định lý 1: duality gap bằng 0 với lớp policy tập trung ngẫu nhiên (Paternain et al. 2019, Thm 1), và gap ≤ O(ε/(1−γ)) với policy tham số hoá ε-phổ quát (Thm 2 của cùng bài). Nêu rõ vì sao không áp dụng thẳng cho policy phân tán (Kahe & Kebriaei 2024).
  0. Định nghĩa ràng buộc và mục tiêu trên **cùng một** đại lượng mà policy tối ưu: return chiết khấu, hoặc trung bình của ca kèm cận cho phần lệch γ. Đây là điểm kiểm độc lập chỉ ra.
  3. Định lý 2: bộ điều khiển chỉ phần I là subgradient đối ngẫu với oracle ε-tối ưu và ước lượng nhiễu. Trung bình các episode (policy trộn theo thời gian) có vi phạm ràng buộc O(1/(ηT)) + O(η) và mục tiêu cách tối ưu ε + O(η). Dựa trên Nedić & Ozdaglar 2009 (SIAM J. Optim. 19(4)), tức primal averaging. Điểm quan trọng: TD3 tất định chỉ cho các đỉnh, còn việc trộn theo thời gian lấp được bao lồi (CAPQL Prop. 3–4).
  4. Định lý 3: ngưỡng hoặc môi trường đổi trong ca, cận theo dõi (dynamic regret) phụ thuộc tổng độ dịch của nghiệm tối ưu.
  5. Tầng UAV: mệnh đề và thí nghiệm, không định lý.
- **Đạt khi:** mỗi chứng minh tự kiểm được từng dòng; mỗi giả định được nêu và có thí nghiệm kiểm nó (ví dụ: đo ε bằng khoảng cách tới ω cố định tốt nhất).

### TIP 11. Pilot v2

- **Task:** config train v2, chạy pilot 300k trên Kaggle (bạn chạy): `ma3_prior` v2 (MLP) và attention v2.
- **Đạt khi:** đường cong học tăng, SLA từng slice và năng lượng có đánh đổi thật (kiểm như phép thử đánh đổi giữa slice).

---

## Completion Reports

(Thêm vào đây theo thứ tự TIP.)
