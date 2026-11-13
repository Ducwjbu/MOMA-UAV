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
| Số UAV (chốt 2026-11-13) | Mỗi ca rút số UAV của đội trong 2–10 khi train. Test trong dải và ở 11–12; chỉ test 13–15 nếu 11–12 đạt tiêu chí của TIP 11 | Dải khác: đổi `fleet` trong config |
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

### TIP 4. Ca trực, thay ca bằng UAV dự phòng, trạm sạc

Khoá `shift` (tắt khi vắng). Một episode là một cửa sổ 4 giờ (240 bước) của một chiến dịch đang chạy.

- **Vị trí phục vụ và thân máy tách nhau.** `num_uavs` là số vị trí phục vụ tối đa (10). Mỗi vị trí là một khe của quan sát và hành động, như hiện nay. Mỗi ca rút S trong `n_serving` (2–10) vị trí được dùng, chọn ngẫu nhiên trong các khe. Đội có S + ceil(`spares_per_uav` × S) thân máy. Policy điều khiển vị trí phục vụ; thân máy đang bay ra, bay về, đổi pin hay chờ ở mặt đất thì env tự điều khiển và không phục vụ.
- **Thay ca (relay).**
  - Khi pin của vị trí i còn bằng năng lượng bay về cộng năng lượng chờ người thay bay tới (công suất tối đa × (thời gian bay ra + 1 bước)), env phái một thân máy dự phòng sẵn sàng bay tới vị trí đó.
  - Người thay tới nơi thì nhận vị trí: vị trí giữ nguyên chỗ đứng, pin bằng pin của người thay. Thân máy cũ bay về Home Point.
  - Không có dự phòng sẵn sàng, hoặc người thay chưa tới khi pin chạm mức bay về, thì thân máy cũ vẫn về (Low Battery RTH). Vị trí bị bỏ trống (gap) cho tới khi có thân máy tới.
  - Thời gian và năng lượng của chặng bay ra, bay về tính bằng `_home_leg_energy` (gió, airspeed). Chặng bay ra tính tới chỗ của vị trí lúc phái đi.
- **Đổi pin.** Thân máy về tới nơi, đổi pin nóng mất `swap_steps` = 1 bước (60 s, Lee et al. 2015). Cặp pin cũ vào kho; thân máy lấy cặp đầy nhất nếu cặp đó đạt `launch_soc` (0.9), nếu không thì chờ.
- **Trạm sạc (BS60, 220 V).**
  - Kho chung có `pairs_per_station` = 4 cặp mỗi trạm, đầy lúc đầu ca.
  - Mỗi trạm sạc một cặp tại một thời điểm, cặp còn nhiều pin nhất trước.
  - Đường sạc CC–CV khớp số của DJI: 20→90% trong 30 phút, đầy trong 60 phút. Giả định "đầy" tính từ 0%: 2.33%/phút tới 90%, rồi 0.47%/phút tới 100%.
  - Số trạm: `stations_per_uav` × S (làm tròn lên) hoặc `station_count` cố định. Điện không giới hạn.
- **Đầu ca.** Pin của S vị trí so le đều trong khoảng dùng được (pha (k + U)/S), như một chiến dịch đang chạy. Dự phòng ở mặt đất, pin đầy.
- **Kết thúc ca là terminal** (`done` ở bước 240). Luật kết thúc khi một UAV chạm năng lượng bay về của `return_home` không áp dụng nữa. Quan sát thêm 1 chiều: thời gian còn lại của ca.
- **Mục tiêu năng lượng** vẫn là trung bình trên các vị trí đang phục vụ (phần policy điều khiển). Năng lượng của các chặng bay ra, bay về ghi riêng trong `info`.
- Prior đội hình gán lại đỉnh khi tập vị trí đang phục vụ đổi, như với `fleet`. Không dùng chung với `fleet`.
- **Đạt khi:**
  - (`docs/v2/check_tip4.py`) ca kết thúc đúng bước 240 với `done`;
  - mọi vị trí được dùng có ít nhất một lần thay ca;
  - pin không âm; số cặp pin bảo toàn;
  - với S = 3, đủ dự phòng và trạm thì không có gap;
  - với S = 10 và ít trạm thì xuất hiện chờ pin;
  - `check_v11` không đổi; smoke train v2 chạy hết.

### TIP 5. Tải theo giờ và sự kiện

- **Task:** hệ số tải theo giờ (nguồn: đo lưu lượng di động theo giờ, cần tìm); sự kiện: điểm nóng xuất hiện rồi tan (đám đông tụ ở điểm tập kết), UAV hỏng giữa ca, ngưỡng của nhà mạng đổi. Sự kiện rút từ RNG có seed, ghi trong `info['events']`.
- **Đạt khi:** cùng seed cho cùng chuỗi sự kiện bất kể policy; trình xem hiện sự kiện.

### TIP 6. UE có trọng số và hiệu chỉnh tải

- **Task:** mỗi UE mô phỏng đại diện `w` người dùng (tải và đếm SLA nhân `w`). Hiệu chỉnh tải danh nghĩa để policy tốt đạt các ngưỡng (Notion 3.1, điều kiện 1).
- **Tải theo kịch bản, không theo đội:** số UE và tải cố định theo vùng, số UAV thay đổi, nên đánh đổi số UAV–SLA–năng lượng là có thật. HV báo cáo theo từng N, không gộp.
- **Đạt khi:** ở tải danh nghĩa, tam giác đứng yên với phân bổ hợp lý đạt khoảng 0.8–0.9 mỗi slice khẩn cấp (để ngưỡng 0.95 khó nhưng với tới được); bảng hiệu chỉnh ghi trong `sources.md`.

### TIP 7. Prior theo nhu cầu

- **Task:** đỉnh của prior = k-means có trọng số trên bản đồ mật độ (đã có `traffic_map`) thay cho tam giác cố định, gán bằng Hungarian (đã có). Khoá `prior_mode: demand | ngon`.
- **Bắt buộc khi N thay đổi:** đa giác đều bán kính 600 m chỉ gần vị trí phủ tốt ở N = 3 (lệch 0.46% so với 3-means). Ở N = 10 nó để trống giữa vùng và các đỉnh cách nhau 371 m. Prior tính k-means cho đúng số UAV đang phục vụ, tính lại khi số đó đổi trong ca (đổi pin, hỏng).
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
- **Thí nghiệm tổng quát theo N (sau pilot, train 2–10):** đánh giá ở N = 2, 4, 6, 8, 10 (trong dải) và 11, 12 (ngoài dải). Tiêu chí "vẫn tốt" chốt trước khi chạy: ở mỗi N ngoài dải, HV của policy attention lớn hơn HV của baseline không train (UAV đứng yên ở vị trí k-means, cùng N, cùng episode), cận dưới CI của hiệu theo cặp > 0. Đạt cả 11 và 12 thì test thêm 13–15, không đạt thì dừng ở 12 và báo cáo đúng như vậy. Mạng MLP chỉ chạy được tới 10 (số khe cố định).
- **Chi phí đã đo** (v2, hành động ngẫu nhiên): một bước env 14.5 ms ở N = 2, 15.9 ms ở N = 3, 22.1 ms ở N = 6, 29.7 ms ở N = 10. Replay buffer của MLP 10 khe khoảng 5–6 GB mỗi run (3 UAV: khoảng 1 GB). Run dài cần `--resume` trên Kaggle.

---

## Completion Reports

(Thêm vào đây theo thứ tự TIP.)

### TIP 0. Bộ kiểm v1.1 bit-for-bit (2026-11-13)

- **STATUS:** xong.
- **FILES CHANGED:** `docs/v2/check_v11.py` (mới).
- **TEST RESULTS:**
  - Tham chiếu ghi từ một worktree sạch của commit `ee68411` (v1.1 cộng các sửa tài liệu).
  - Code hiện tại so với tham chiếu: env 4224 mảng, 0 lệch; train 96 tensor (smoke `ma3_prior` và `gpi_pd`), 0 lệch.
  - Phần env chạy hai lần trên cùng code: 0 lệch.
  - Đổi `noise_figure_db` từ 9.0 lên 9.1 trong `ma3.yaml`: script báo lệch từ bước đầu, exit 1. Config đã trả về nguyên trạng.
- **ISSUES:** tham chiếu phụ thuộc máy (CPU, thư viện), nên nằm trong `runs/` và không commit. Máy khác phải `record` trên code v1.1 trước.
- **DEVIATIONS:**
  - Episode lấy seed validation 100000+, như `eval_omega`.
  - Hành động lấy từ một `RandomState` riêng, để không đụng RNG của env.
  - Train dùng 1 luồng torch, cho kết quả ổn định.
- **SUGGESTIONS:** chạy `check` sau mỗi TIP. Mất khoảng 4 phút cả hai phần; chỉ phần env khoảng 45 giây.

### TIP 1. Ngưỡng SLA theo 5QI và mục tiêu theo slice (2026-11-13)

- **STATUS:** xong phần code và kiểm; tốc độ của slice dữ liệu là giá trị tạm.
- **FILES CHANGED:**
  - `environment.py`: mục `objectives`, `_sla_margin`, `_sla_score`, `ue.failure_rate` tính chính xác bằng `expm1/log1p`, `objective_names`.
  - `train_moma_ac.py`: số mục tiêu và tên lấy từ env; `_hv_nd`, `hv_ref_point`.
  - `agents_moma_ac.py`: chỉ đổi dòng in khi số mục tiêu khác 3.
  - Mới: `config/environment/v2.yaml`, `config/train/moma_ac_v2_prior.yaml`, `docs/v2/check_tip1.py`, `docs/v2/sources.md`.
  - `kaggle/build_bundle.py` thêm hai config v2.
- **TEST RESULTS:**
  - **Kiểm β lớn:** với β = 1e12, mục tiêu `sla_*` trùng tỉ lệ nhị phân trong `info` ở mọi bước (5 episode validation, 90 bước). Lệch tối đa 1.4e-8, do reward vector lưu float32. Với β = 4: mục tiêu 0.115 / 0.064 / 0.173 so với tỉ lệ nhị phân 0.116 / 0.065 / 0.173 (hành động ngẫu nhiên).
  - **HV n chiều:** trùng `_hv_3d` từng bit trên 400 front của `results-v1.1` (thêm chiều thứ tư hằng số). Với 3 tập 25 điểm 4 chiều ngẫu nhiên, nằm trong khoảng tin cậy 95% của Monte Carlo (2 triệu mẫu).
  - **Smoke train v2:** `moma_ac_v2_prior` với smoke settings chạy hết 1500 bước, có eval và checkpoint. CSV có cột theo tên mục tiêu.
  - **v1.1 không đổi:** `check_v11.py check` cho 0 lệch ở cả env và train.
- **ISSUES:**
  - Tốc độ video của lực lượng cứu hộ (2 Mbps) là giá trị tạm lấy từ TS 22.125. Cần một nguồn tốt hơn.
  - Bảng 5QI đọc từ bản dựng lại TS 23.501 v20.2.0, chưa đối chiếu PDF chính thức.
  - Tải mới (31 UE dữ liệu × 2 Mbps) nhẹ hơn nhiều so với v1.1. Số UE và tải hiệu chỉnh ở TIP 6.
  - `eval_checkpoint.py`, `aggregate_seeds.py`, `eval_controller.py`, `plot_results.py` vẫn giả định 3 mục tiêu. Tổng quát hoá ở TIP 8.
- **DEVIATIONS:**
  - Slice dữ liệu dùng 5QI 67 (video khẩn cấp, PER 1e-3) thay vì 5QI 70 (PER 1e-6). Lý do: env chỉ mô hình HARQ, PER dư của eMBB là 1e-4, nên 1e-6 không bao giờ đạt (Notion 3.1).
  - Slice giữ tên nội bộ `embb` / `urllc` / `mmtc`; ý nghĩa theo câu chuyện (b) ghi ở đầu `v2.yaml`.
  - mMTC giữ nguyên ngưỡng v1.1.
  - Slice không có tải ở một bước thì mục tiêu bằng 1.
  - URLLC: biên tin cậy lấy bằng 2 lần truyền (PER dư 1e-10).
- **SUGGESTIONS:** chọn β bằng một phép thử nhỏ ở TIP 11: β nhỏ cho gradient mượt nhưng xa chỉ số thật; β lớn thì ngược lại.

### TIP 2. Hành động theo slice và bộ lập lịch bên trong (2026-11-13)

- **STATUS:** xong.
- **FILES CHANGED:**
  - `environment.py`: khoá `action.mode`, `act_dim`, `_provisional_rb_rates`, `_scheduler_target_loads`, `_waterfill`, `_schedule_slices`, `_allocate_rbs_slices`.
  - `train_moma_ac.py`: `act_dim` lấy từ env.
  - `config/environment/v2.yaml`: `action.mode: slice_share`, `per_slice_queues: true`.
  - Mới: `docs/v2/check_tip2.py`.
  - `docs/v2/check_tip1.py`: hành động theo `act_dim`.
- **TEST RESULTS:**
  - **Tam giác đứng yên, chia đều, 10 episode validation (`check_tip2.py`):** SLA eMBB / URLLC / mMTC từ 0.212 / 0.148 / 0.450 (9 phần DA đều; 17% RB bỏ trống ở mọi bước có tải) lên 0.327 / 0.353 / 0.450, không còn RB bỏ trống. PASS.
  - **Bản đầu không có tải mục tiêu** (cấp đúng bằng tải): eMBB giảm còn 0.131, vì hàng đợi chạy gần tải 1 nên drop trung vị 9.3e-4, sát ngân sách 1e-3. Đã sửa bằng `_scheduler_target_loads`.
  - **v1.1 không đổi:** `check_v11.py` 0 lệch, env và train.
  - **Kiểm khác:** smoke train v2 (hành động 7 chiều mỗi UAV) chạy hết. Kiểm TIP 1 vẫn PASS.
  - **Thời gian mỗi bước:** 18.6 ms, so với 17.6 ms khi không có bộ lập lịch và 15.8 ms với `ma3.yaml`.
- **ISSUES:**
  - mMTC bằng nhau ở hai chế độ: bị giới hạn bởi độ phủ, không phải băng thông.
  - Tốc độ mỗi RB ước lượng trước khi chia RB, giả sử mọi UAV bận dùng hết RB. Giả định này đúng với bộ lập lịch, trừ UAV mà mọi UE có tốc độ 0. Khi đó nhiễu thật nhỏ hơn ước lượng một chút.
  - Bộ lập lịch không tính giới hạn backhaul (v2 gần như không chạm, như v1.1).
  - Các công cụ v1.1 đọc hành động 13 chiều (`eval_trajectories.py`, `eval_heuristics.py`, `visualize/`) chưa hỗ trợ 7 chiều. Làm ở TIP 8.
- **DEVIATIONS:**
  - Plan ghi "proportional fair"; bản cài là nghiệm của proportional fair khi tốc độ mỗi RB cố định trong bước, có trần nhu cầu (water-filling số RB). Không mô phỏng từng TTI.
  - Thêm tải mục tiêu mỗi slice (plan không có), suy từ SLA của slice.
  - Hàng đợi riêng mỗi slice dùng khoá `per_slice_queues` có sẵn.
- **SUGGESTIONS:** heuristic và baseline tối ưu (TIP 9) nên dùng cùng bộ lập lịch này, để so sánh chỉ khác ở vị trí và tỉ lệ slice.

### TIP 4. Ca trực, thay ca bằng UAV dự phòng, trạm sạc (2026-11-13)

- **STATUS:** xong.
- **FILES CHANGED:**
  - `environment.py`: khoá `shift`; `_home_leg` (năng lượng và thời gian, `_home_leg_energy` gọi nó, số học không đổi); `_hover_power`, `_leave_threshold`, `_relay_threshold`, `_start_shift`, `_charge_soc`, `_send_home`, `_dispatch`, `_shift_update`; quan sát thêm thời gian còn lại; `done` ở cuối ca; `info` thêm `n_serving_target`, `n_serving_step`, `relays`, `gaps`, `waiting_battery`, `transit_energy_j`, `depot_mean_soc`.
  - `prior.py`: gán lại đỉnh khi tập vị trí đang phục vụ đổi trong ca; tập rỗng trả về `{}`.
  - `config/environment/v2.yaml`: `num_uavs: 10` (vị trí phục vụ), mục `shift`.
  - `config/train/moma_ac_v2_prior.yaml`: `max_steps_per_episode` và `eval_steps` 240.
  - Mới: `docs/v2/check_tip4.py`. `check_tip1.py`, `check_tip2.py` cố định 3 UAV, không có ca, như lúc viết.
- **TEST RESULTS:**
  - **`check_tip4.py`, 2 ca validation mỗi bộ, các vị trí đứng yên ở chỗ xuất phát, công suất 1 (PASS):**

    | Bộ | Thay ca | Tỉ lệ vị trí trống | Bước chờ pin (thân máy) | Đổi pin mỗi thân máy |
    |---|---|---|---|---|
    | A: S = 3, 6 BS60, 3 dự phòng | 86 | 0.000 | 0 | 7.1 |
    | B: S = 10, 3 BS60 | 34 | 0.684 | 5143 | 3.0 |
    | v2: S = 3 | 83 | 0.001 | 0 | 8.4 |
    | v2: S = 10 | 278 | 0.000 | 0 | 9.1 |
    | S = 3, 1 BS60 mỗi vị trí | 40 | 0.237 | 732 | 6.4 |
    | S = 10, 1 BS60 mỗi vị trí | 127 | 0.241 | 2007 | 7.0 |

    Mọi bộ: ca kết thúc đúng bước 240, pin không âm, không chặng về nào thiếu năng lượng, số cặp pin bảo toàn, mọi vị trí được thay ca ít nhất một lần.
  - **Mỗi vị trí được thay khoảng 14 lần mỗi ca** (khoảng 17 phút một lần khi treo ở công suất tối đa), không phải 25 phút như ước lượng trước: ngưỡng gọi người thay giữ lại năng lượng cho chặng về, cho người thay bay ra và một bước dự phòng.
  - **Quét trang bị** (S = 3 và 10, trống / bước chờ): 1 trạm mỗi vị trí trống 24–26%; 1.5 trạm 2–3%; 2 trạm dưới 1%. Với 2 trạm, 0.25 dự phòng trống 1%, 0.5 dự phòng 0–0.1%.
  - **v1.1 không đổi:** `check_v11.py` 0 lệch (env 4224 mảng, train 96 tensor). `check_tip1.py`, `check_tip2.py` PASS với đúng số cũ.
  - **Smoke train v2** (10 khe, hành động 7 chiều, 4 mục tiêu) chạy hết với mạng MLP và attention.
  - **Thời gian mỗi bước** (đứng yên, 10 khe): 24 ms với S = 3, 33 ms với S = 10.
- **ISSUES:**
  - Mặc định trang bị (2 BS60, 0.5 dự phòng mỗi vị trí) là thiết kế, đo để gần như không hở; chưa có nguồn cho trang bị thật của đội cứu hộ.
  - Chặng bay ra tính tới chỗ của vị trí lúc phái đi; người thay nhận vị trí ở chỗ hiện tại (vị trí chỉ dịch quanh đỉnh của prior).
  - Năng lượng các chặng bay ra, bay về không nằm trong mục tiêu năng lượng (policy không điều khiển chúng), chỉ ghi trong `info`.
  - Nếu mọi vị trí cùng trống, mục tiêu năng lượng bằng 0 (tốt nhất) ở bước đó. Với trang bị mặc định thì không xảy ra.
  - Eval 240 bước × 30 ω × 3 episode khá tốn (khoảng 7 phút mỗi lần eval). Xem lại ở TIP 8/11.
  - Replay buffer với 10 khe: khoảng 2.8 GB cho 500k mẫu.
  - Prior vẫn là đa giác đều r = 600 m (chỉ hợp với 3). TIP 7 thay bằng k-means theo số vị trí đang phục vụ.
- **DEVIATIONS:**
  - Vị trí phục vụ (khe của policy) tách khỏi thân máy. Thay ca là đổi thân máy dưới một vị trí, nên quan sát và hành động chỉ cần 10 khe, không cần thêm khe cho dự phòng.
  - Không dùng `fleet` (plan cũ); `shift` tự quản lý vị trí đang phục vụ, và không dùng chung với `fleet`.
  - Quan sát chỉ thêm thời gian còn lại của ca; trạng thái của vị trí (đang phục vụ hay trống) đã thể hiện bằng công suất 0, như với `fleet`.
  - UAV về sớm hơn một bước so với ngưỡng năng lượng bay về, vì env kiểm mỗi bước 60 s còn M300 kiểm liên tục. Bản đầu về đúng ngưỡng, và 14–28 chặng về mỗi ca bị thiếu năng lượng.
- **SUGGESTIONS:** TIP 5 có thể thêm sự kiện "trạm sạc hỏng" hoặc "thiếu pin" bằng cách đổi số trạm giữa ca: env đã đo được hậu quả (vị trí trống, bước chờ pin).

### TIP 4b. Mô hình năng lượng theo M300 và ngưỡng thay ca sát thực tế (2026-11-13)

- **STATUS:** xong. Bạn hỏi 17 phút giữa hai lần thay ca có ít quá không. Có hai nguyên nhân: mô hình năng lượng bi quan hơn M300 thật, và luật thay ca giữ dự phòng quá dày.
- **FILES CHANGED:**
  - `config/environment/v2.yaml`: `energy_models` hiệu chỉnh theo M300 ở 9 kg; `shift.relay_power_margin`, `shift.power_smoothing`; `stations_per_uav` 1.5.
  - `environment.py`:
    - `_step_energy_bound` (mức tối đa một bước có thể tiêu, chính xác thay cho `max_total_power × T_L`);
    - `_leave_threshold` dùng nó;
    - `_relay_threshold` dùng công suất gần đây của vị trí (trung bình mũ trong `shift_state['power']`) cộng 10%.
- **TEST RESULTS:**
  - **Năng lượng:** treo cộng radio tối đa 1189 W (trước 1500 W); pin đầy treo được 27.7 phút (trước 21.9).
  - **Ngưỡng ở giữa vùng (theo phần pin):**

    | Ngưỡng | Trước | Sau |
    |---|---|---|
    | Chặng về | — | 3.5% |
    | Phải về | 11% | 7.5% |
    | Gọi người thay | 41% | 23% |

  - **`check_tip4.py` PASS:**
    - Khoảng thay ca khi treo ở công suất tối đa từ khoảng 17 phút lên khoảng 26 phút (S = 3: 28 lần thay mỗi ca cho 3 vị trí).
    - Với 1 BS60 mỗi vị trí, tỉ lệ vị trí trống từ 24% xuống 5%.
  - **Quét trang bị:** 1.5 BS60 và 0.5 dự phòng mỗi vị trí cho 0% trống ở S = 3 và 10, nên mặc định đổi từ 2 xuống 1.5.
  - **Kiểm khác:** v1.1 không đổi (`check_v11.py` 0 lệch). TIP 1, TIP 2 PASS với số cũ.
- **ISSUES:**
  - Hiệu chỉnh chỉ khớp ở 9 kg: ba mốc của DJI (55 phút không tải, 43 phút với H20T, 31 phút ở 9 kg) không khớp cùng lúc với mô hình treo của Zeng.
  - Con số 31 phút là của bên thứ ba và được coi là thời gian treo; nếu DJI đo khi bay, công suất treo thật cao hơn.
  - Không trừ thêm hệ số dung lượng thực tế: các ngưỡng đã giữ năng lượng cho chặng về, và gió được mô phỏng riêng.
  - Công suất vượt dự báo chỉ làm người thay tới muộn, vì UAV cũ vẫn về ở ngưỡng an toàn, nên vị trí trống một lúc. Ở bộ kiểm (công suất không đổi) điều này không xảy ra.
- **DEVIATIONS:** khối lượng trạm phát chưa có nguồn; giả định dùng hết tải 2.7 kg.

### TIP 7. Prior theo nhu cầu (2026-11-14)

- **STATUS:** xong.
- **FILES CHANGED:**
  - `prior.py`: `demand_points`, `weighted_kmeans`, `demand_vertices`; `assign_vertices` và `attach` nhận `mode` (`ngon` | `demand`); `env.prior_targets`.
  - `train_moma_ac.py`: khoá train `prior_mode` (mặc định `ngon`).
  - `config/train/moma_ac_v2_prior.yaml`: `prior_mode: demand`.
  - Mới: `docs/v2/check_tip7.py`.
- **TEST RESULTS:**
  - **`check_tip7.py`** (prior không có phần dư, công suất 1, chia slice đều, 3 ca validation × 60 bước, từ bước 6). PASS theo tiêu chí đặt trước: coverage trung bình theo S 0.625 với demand, 0.572 với ngon; SLA trung bình 0.491 / 0.451.

    | S | Coverage ngon → demand | SLA trung bình ngon → demand |
    |---|---|---|
    | 2 | 0.376 → 0.414 | 0.178 → 0.209 |
    | 3 | 0.517 → 0.528 | 0.370 → 0.307 |
    | 6 | 0.662 → 0.746 | 0.573 → 0.655 |
    | 10 | 0.732 → 0.814 | 0.685 → 0.792 |

  - **Thời gian k-means:** 0.06 s (S = 2) đến 0.33 s (S = 10), tính lúc reset và khi số vị trí đang phục vụ đổi.
  - **v1.1 không đổi** (`check_v11.py`). Smoke train v2 với `prior_mode: demand` chạy hết.
- **ISSUES:**
  - Ở S = 3, k-means phủ hơn nhưng SLA URLLC thấp hơn tam giác (0.179 so với 0.380). Coverage dùng khoảng cách bình phương làm đại diện, không theo SINR, và RSU nằm trên lưới đường: tam giác r = 600 m tình cờ hợp với lưới RSU. Policy học phần dư có thể sửa chỗ này; nếu không, có thể thay khoảng cách bằng suy hao theo mô hình A2G.
  - k-means dựa trên bản đồ mật độ của episode, tức là giả định nhà mạng biết phân bố tải dự kiến (ở mức non-RT RIC), không phải vị trí từng UE.
  - Đỉnh ở độ cao tối đa như prior cũ.
- **DEVIATIONS:**
  - Plan ghi "trên bản đồ có điểm nóng lệch tâm"; bản kiểm dùng bản đồ thật của các episode validation, với S từ 2 đến 10.
  - Trọng số theo slice (mỗi slice nặng như nhau) không có trong plan.
- **SUGGESTIONS:** baseline "đứng yên ở đỉnh k-means" của thí nghiệm tổng quát theo N (TIP 11) chính là `check_tip7.py` ở chế độ demand với hành động 0; nên đưa nó vào `eval_heuristics.py` ở TIP 8.

### Vùng ngẫu nhiên theo episode, dân số dừng (2026-11-14, trước TIP 6)

- **STATUS:** xong. Bạn hỏi map có phải chỉ một không: chiều cao nhà, bản đồ mật độ, gió và UE đã đổi theo episode, nhưng bố cục phố, vị trí RSU và Home Point cố định. Bạn đồng ý giữ 2 km × 2 km (10 UAV vẫn thưa hơn mạng macro UMa) và làm theo đề xuất.
- **FILES CHANGED:**
  - `city.py`: `shift` (dời lưới), `nearest_crossing`, `in_street`, `_building_at_grid`; LoS và `building_at` làm trên toạ độ lưới.
  - `environment.py`:
    - `city.random_offset`, `rsu.random_offset`, `return_home.random_edge` (`_draw_home`, gateway backhaul đi theo);
    - `ue_dynamics.stationary`;
    - bám phố của UE và RSU qua `City.nearest_crossing` / `in_street`.
  - `config/environment/v2.yaml`: bật cả bốn khoá; bỏ nhãn tạm của 2 Mbps.
- **TEST RESULTS:**
  - **Lưới dời:**
    - LoS numba và numpy khớp trên 100k đoạn (5 lưới dời);
    - lưới dời cho đúng LoS và ô nhà của lưới gốc dịch tương ứng (60k đoạn, 0 lệch);
    - trong nhà khi và chỉ khi không nằm trên phố (100k điểm).
  - **Dân số dừng:** trung bình UE đến 90.0 (10 episode × 31 bước; mục tiêu 91), không còn trôi.
  - **v1.1 không đổi.** TIP 1, 2, 4, 7 PASS trên v2 mới.
  - **TIP 7 với RSU ngẫu nhiên:** điểm yếu URLLC ở S = 3 biến mất (k-means 0.380, tam giác 0.178; trước đó 0.179 và 0.380). Tam giác trước đây thắng vì tình cờ khớp lưới RSU cố định. Giờ k-means hơn ở mọi S: SLA trung bình 0.534 so với 0.459, coverage 0.640 so với 0.589.
  - **Chẩn đoán cho TIP 6** (k-means đứng yên, chia slice đều, 3 ca × 60 bước):
    - eMBB và mMTC trượt SLA gần như chỉ vì không được phủ (eMBB 12–53%, mMTC 27–68% số UE-bước, S = 10 → 2);
    - chỉ URLLC trượt vì tốc độ, trễ, lỗi gói, chủ yếu khi S nhỏ;
    - năng lực hiếm khi là ràng buộc với tải hiện tại.
- **ISSUES:**
  - Kết quả các TIP trước trên v2 đổi số vì vùng ngẫu nhiên (bảng trong các Completion Report là số lúc đó).
  - Video nhiệm vụ thật chủ yếu là đường lên, env chỉ có đường xuống.

### TIP 6. Hiệu chỉnh tải (2026-11-14)

- **STATUS:** xong một phần.
  - Dân số dừng: xong (commit trước).
  - Hiệu chỉnh tải: xong, nhưng kiểm (c) trượt sát nút (xem TEST RESULTS).
  - Bạn chọn phương án B: tăng tải eMBB để năng lực thật sự là ràng buộc ở đội cỡ giữa.
- **FILES CHANGED:**
  - `environment.py`:
    - `traffic_patterns.embb.streams_per_ue` (`self.embb_streams`, `ue.streams`): tải và tốc độ yêu cầu nhân theo;
    - `_required_rate` dùng cho SLA, độ thoả mãn tốc độ và fairness `min_rate`.
  - `config/environment/v2.yaml`: `streams_per_ue: 2.6`.
  - Mới: `docs/v2/check_tip6.py` (`calibrate`, `check`).
  - `docs/v2/check_tip1.py`: KPI tốc độ theo `_required_rate`.
- **TEST RESULTS:**
  - **Bản đầu (tải bằng năng lực của 6 UAV, như cách v1.1 làm cho 3 UAV):** 17 luồng mỗi UE. Bản này PASS bộ tiêu chí lúc đó nhưng không dùng được: SLA eMBB chỉ 0.07 ở S = 6 và 0.17 ở S = 10 (năng lực dư gần gấp đôi). Tải dồn vào từng UE (34 Mbps mỗi UE), UE ở rìa ô không nhận đủ dù chia slice thế nào. Năng lực tính theo tốc độ trung bình mỗi RB bỏ qua chỗ tải nằm.
  - **Quét lại** theo mức 0.8 của plan (covered = trong các UE eMBB đã được phủ):

    | w | 1.0 | 2.0 | 2.5 | 2.6 | 2.7 | 3.0 |
    |---|---|---|---|---|---|---|
    | Đạt SLA (covered) | 0.993 | 0.899 | 0.816 | 0.801 | 0.785 | 0.722 |
    | Trượt vì tốc độ (covered) | 0.003 | 0.054 | 0.094 | 0.100 | 0.107 | 0.132 |

    Chọn 2.6.
  - **`check_tip6.py check` với 2.6:**

    | S | Năng lực Mbps | Tải Mbps | eMBB covered đạt / trượt vì tốc độ | SLA eMBB / URLLC / mMTC |
    |---|---|---|---|---|
    | 2 | 352 | 400 | 0.247 / 0.425 | 0.129 / 0.031 / 0.344 |
    | 4 | 856 | 414 | 0.583 / 0.212 | 0.430 / 0.840 / 0.510 |
    | 6 | 1335 | 417 | 0.801 / 0.100 | 0.610 / 0.582 / 0.595 |
    | 8 | 1615 | 413 | 0.776 / 0.099 | 0.674 / 0.797 / 0.705 |
    | 10 | 2205 | 403 | 0.955 / 0.028 | 0.875 / 0.951 / 0.727 |

    - (a) dân số 90.2: ok.
    - (b) 0.801 nằm trong [0.80, 0.90]: ok.
    - (d) chia slice có tác dụng: (0.6, 0.2, 0.2) cho eMBB 0.680, URLLC 0.536; (0.2, 0.6, 0.2) cho eMBB 0.525, URLLC 0.790: ok.
    - **(c) FAIL sát nút:** 246/2468 = 0.0997 < 0.10. Hai tiêu chí đặt trước (b ≥ 0.80 và c ≥ 0.10) không cùng thoả trên lưới 0.1 (2.7 cho c 0.107 nhưng b 0.785). Không đổi ngưỡng sau khi thấy số; chờ bạn quyết.
  - **Kiểm khác:** v1.1 không đổi; TIP 1, 2, 4, 7 chạy lại trên v2 mới (xem commit).
- **ISSUES:**
  - Mức tải là thiết kế (không có nguồn cho số đội cứu hộ trên mỗi km²).
  - S = 8 có tỉ lệ đạt (0.776) thấp hơn S = 6 (0.801): k-means đặt UAV theo cả ba slice, và 3 episode là mẫu nhỏ.
  - Tỉ số tải / năng lực ở S = 6 chỉ 0.31: năng lực trung bình không phải thước đo đúng, ràng buộc nằm ở UE sóng yếu.
- **DEVIATIONS:**
  - Plan ghi "UE có trọng số (tải và đếm SLA nhân w)". Bản cài nhân tải và tốc độ yêu cầu; mỗi UE vẫn đếm một lần trong tỉ lệ SLA của slice, vì mọi UE eMBB cùng w nên trung bình không đổi.
  - Ngưỡng hiệu chỉnh lúc đầu là 0.75 (lệch plan), sửa về 0.80 của plan.
