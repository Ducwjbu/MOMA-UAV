# Nguồn cho các tham số của môi trường v2

Mỗi tham số mới của `config/environment/v2.yaml` có một dòng ở đây: giá trị, nguồn đầy đủ, phần cụ thể đã dùng, và trạng thái (đã đọc bản chính thức hay chưa). Tham số giữ từ v1.1 xem comment trong `ma3.yaml` và CLAUDE.md.

## TIP 1. Ngưỡng SLA và mục tiêu theo slice

| Tham số | Giá trị | Nguồn | Phần dùng | Trạng thái |
|---|---|---|---|---|
| `qos_profiles.embb.max_latency` | 90 ms | 3GPP TS 23.501, "System architecture for the 5G System (5GS)", Table 5.7.4-1 (standardized 5QI to QoS characteristics mapping) | 5QI 67 (Mission Critical Video user plane): PDB 100 ms; ghi chú của bảng: 10 ms cho phần mạng lõi với các 5QI mission critical, nên còn 90 ms cho vô tuyến và hàng đợi | Đọc từ bản dựng lại v20.2.0 (whatthespec.net), chưa đối chiếu PDF chính thức |
| `qos_profiles.embb.min_reliability` | 0.999 | Như trên | 5QI 67: PER 1e-3 | Như trên |
| `qos_profiles.embb.min_rate`, `traffic_patterns.embb.peak_rate` | 2 Mbps | 3GPP TS 22.125, "Unmanned Aerial System (UAS) support in 3GPP", Table 7.2-2 | Tốc độ video 480p cho điều khiển UAV. **Giá trị tạm**: chưa có nguồn cho tốc độ video của lực lượng cứu hộ | Bản scrape (itecspec.com) |
| `traffic_patterns.embb.activity_factor` | 1.0 | Thiết kế | Video phát liên tục; số UE và tải hiệu chỉnh ở TIP 6 | — |
| `qos_profiles.urllc.max_latency` | 25 ms | TS 23.501 Table 5.7.4-1 | 5QI 84 (Intelligent Transport Systems): PDB 30 ms, ghi chú: 5 ms cho mạng lõi | Bản dựng lại v20.2.0 |
| `qos_profiles.urllc.min_reliability` | 0.99999 | TS 23.501 Table 5.7.4-1; TS 22.261 Table 7.2.3.2-1 | 5QI 84: PER 1e-5; TS 22.261 (road-side infrastructure): 99.999% | Như v1.1 |
| `link_adaptation.urllc.max_transmissions` | 2 | Thiết kế, dựa trên `harq_rtt_ms` 8 ms của v1.1 | Hai lần truyền (16 ms) vừa ngân sách 25 ms. PER dư 1e-10, để lại 1e-5 − 1e-10 cho drop và handover. v1.1 dùng 1 lần, PER dư bằng đúng mục tiêu (`report/05_han_che.md`, hạn chế 11) | — |
| `objectives.names` | `[sla_embb, sla_urllc, sla_mmtc, energy]` | Thiết kế (plan, Q2) | Công bằng chỉ báo cáo, vì tương quan QoS–công bằng 0.98 ở v1.1 | — |
| `objectives.sla_sharpness` | 4 | Thiết kế | UE ở 80% tốc độ yêu cầu được 0.29, ở 120% được 0.67; β → ∞ trùng chỉ số nhị phân (`docs/v2/check_tip1.py`) | — |
| `objectives.sla_tolerance` | 1e-9 | Thiết kế | Dung sai tương đối của phép so ngưỡng | — |

Cách tính SLA mỗi UE (v2): biên nhỏ nhất theo log tự nhiên, `m = min(log(rate / rate yêu cầu), log(ngân sách trễ / trễ), log(ngân sách lỗi / tỉ lệ lỗi))`. UE không được cấp RB thì `m = −∞`. Slice dùng trung bình `sigmoid(β m)` trên các UE có tải. Slice không có UE có tải ở bước đó thì giá trị là 1.

## TIP 2. Hành động theo slice và bộ lập lịch

| Tham số | Giá trị | Nguồn | Phần dùng | Trạng thái |
|---|---|---|---|---|
| `action.mode` | `slice_share` | 3GPP TS 28.541, "Management and orchestration; 5G Network Resource Model (NRM)", RRMPolicyRatio (`rRMPolicyMinRatio`, `rRMPolicyMaxRatio`) | Tỉ lệ tài nguyên theo slice là phần đảm bảo tối thiểu; phần không dùng được chia lại | Như v1.1 (`rb_sharing`) |
| Bộ lập lịch trong slice | Proportional fair có trần nhu cầu | F. Kelly, "Charging and rate control for elastic traffic", European Transactions on Telecommunications 8(1):33–37, 1997 | Tối đa Σ log(thông lượng). Với tốc độ mỗi RB cố định trong bước và trần bằng nhu cầu, nghiệm là số RB bằng nhau có trần (water-filling) | Lập luận trong docstring `_schedule_slices` |
| Thang thời gian | Tỉ lệ slice mỗi bước 60 s, lập lịch bên trong | O-RAN Alliance WG1, "O-RAN Architecture Description" (non-RT RIC: vòng điều khiển ≥ 1 s; near-RT RIC: 10 ms – 1 s) | Đặt hành động ở thang non-RT RIC | **Chưa đọc bản chính thức** (Notion mục 6) |
| `slicing.per_slice_queues` | true | Thiết kế (khoá có từ v1.1), mỗi slice một phần ba buffer | Sửa hạn chế 4: năng lực của mMTC không còn phục vụ gói eMBB | — |
| Tải mục tiêu của bộ lập lịch | Drop = 1/10 ngân sách lỗi | Thiết kế; công thức M/M/1/K của env | eMBB 0.75, URLLC 0.93, mMTC 0.95 | — |

## TIP 4. Ca trực, thay ca, trạm sạc

| Tham số | Giá trị | Nguồn | Phần dùng | Trạng thái |
|---|---|---|---|---|
| `shift.swap_steps` | 1 bước (60 s) | D. Lee, J. Zhou, W. T. Lin, "Autonomous Battery Swapping System for Quadcopter", Proc. ICUAS 2015, pp. 118–124 | Đổi pin tự động trong 60 s | Đọc qua trích dẫn trong Galkin & DaSilva (dưới), chưa đọc bản gốc |
| Đổi pin nóng | Không tắt máy | DJI Store, trang TB60 Intelligent Flight Battery | TB60 hot-swap; M300 bay bằng một cặp TB60 | Trang sản phẩm |
| Thay ca bằng UAV dự phòng | Cơ chế | V. Galkin, L. DaSilva, "UAVs as Mobile Infrastructure: Addressing Battery Lifetime", arXiv:1807.00996, 2018 (preprint) | Sec. IV: UAV dự phòng thay UAV về sạc để phủ liên tục; Sec. V: tổng thời gian gián đoạn dưới 3 phút mỗi 20–25 phút khi đổi pin | Preprint, chưa thấy bản đăng |
| `shift.pairs_per_station` | 4 cặp | DJI Store, trang BS60 Intelligent Battery Station; Advexure (đại lý) | BS60 chứa 8 TB60 | Trang sản phẩm; hướng dẫn sử dụng BS60 (PDF) bị mã hoá, chưa đối chiếu |
| `shift.charge` | 20→90% trong 30 phút, đầy trong 60 phút (220 V) | DJI Store; Advexure | "approximately 60 minutes to fully charge two TB60"; 20–90% trong 30 phút; 110 V: 70 và 40 phút. Giả định "đầy" tính từ 0% | Như trên. B&H ghi "70 min (1 x TB60)" ở 110 V, mâu thuẫn với các nguồn khác |
| Một cặp tại một thời điểm, cặp đầy nhất trước | Cơ chế của kho | Advexure; Heliguy, "How many batteries can the M300 Battery Station (BS60) charge at the same time?" | Sạc 2 TB60 cùng lúc; "selects the battery with the highest battery level to be recharged first" | Trang đại lý |
| `shift.launch_soc` | 0.9 | Thiết kế | Hết đoạn sạc dòng không đổi (CC); đoạn 90–100% chậm gấp 5 lần | — |
| `shift.stations_per_uav` | 2 BS60 mỗi vị trí | Thiết kế, đo bằng `docs/v2/check_tip4.py` | 1 trạm mỗi vị trí: vị trí trống khoảng 25% thời gian (S = 3 và 10); 1.5: 2–3%; 2: dưới 1% | — |
| `shift.spares_per_uav` | 0.5 | Thiết kế, cùng script | Với 2 trạm mỗi vị trí: 0.25 dự phòng → trống 1%, 0.5 → 0.1% (S = 3), 0 (S = 10) | — |
| `shift.n_serving` | 2–10 | Quyết định 2026-11-13 (`plan.md`) | Test thêm 11–12, rồi 13–15 nếu đạt | — |
| `shift.duration_steps` | 240 (4 giờ) | Thiết kế | Đủ để mỗi vị trí được thay ca khoảng 14 lần và kho pin kịp hụt nếu thiếu trạm | — |
