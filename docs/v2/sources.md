# Nguồn cho các tham số của môi trường v2

Mỗi tham số mới của `config/environment/v2.yaml` có một dòng ở đây: giá trị, nguồn đầy đủ, phần cụ thể đã dùng, và trạng thái (đã đọc bản chính thức hay chưa). Tham số giữ từ v1.1 xem comment trong `ma3.yaml` và CLAUDE.md.

## TIP 1. Ngưỡng SLA và mục tiêu theo slice

| Tham số | Giá trị | Nguồn | Phần dùng | Trạng thái |
|---|---|---|---|---|
| `qos_profiles.embb.max_latency` | 90 ms | 3GPP TS 23.501, "System architecture for the 5G System (5GS)", Table 5.7.4-1 (standardized 5QI to QoS characteristics mapping) | 5QI 67 (Mission Critical Video user plane): PDB 100 ms; ghi chú của bảng: 10 ms cho phần mạng lõi với các 5QI mission critical, nên còn 90 ms cho vô tuyến và hàng đợi | Đọc từ bản dựng lại v20.2.0 (whatthespec.net), chưa đối chiếu PDF chính thức |
| `qos_profiles.embb.min_reliability` | 0.999 | Như trên | 5QI 67: PER 1e-3 | Như trên |
| `qos_profiles.embb.min_rate`, `traffic_patterns.embb.peak_rate` | 2 Mbps | 3GPP TS 22.125, "Unmanned Aerial System (UAS) support in 3GPP", Table 7.2-2 | Tốc độ video 480p cho điều khiển UAV. Nguồn cho video cứu hộ: xem mục "Vùng phục vụ ngẫu nhiên" bên dưới (150 kbps – 5 Mbps; 1.5 Mb/s) | Bản scrape (itecspec.com) |
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
| `shift.stations_per_uav` | 1.5 BS60 mỗi vị trí | Thiết kế, đo bằng `docs/v2/check_tip4.py` (mô hình năng lượng TIP 4b) | 1 trạm mỗi vị trí: vị trí trống khoảng 5% thời gian (S = 3 và 10); 1.5 và 2: 0% | — |
| `shift.spares_per_uav` | 0.5 | Thiết kế, cùng script | Với 1.5 trạm mỗi vị trí: 0.25 dự phòng → trống 1.9% (S = 3), 0 (S = 10); 0.5 → 0 | — |
| `shift.relay_power_margin`, `shift.power_smoothing` | 0.1, 0.5 | Thiết kế | Ngưỡng gọi người thay dùng công suất gần đây (trung bình mũ) + 10%; ngưỡng phải về dùng mức tối đa một bước có thể tiêu (bay ngang nhanh nhất gió cho phép, radio tối đa, leo hết dải độ cao) | — |
| `shift.n_serving` | 2–10 | Quyết định 2026-11-13 (`plan.md`) | Test thêm 11–12, rồi 13–15 nếu đạt | — |
| `shift.duration_steps` | 240 (4 giờ) | Thiết kế | Mỗi vị trí được thay ca khoảng 9 lần (treo ở công suất tối đa), kho pin kịp hụt nếu thiếu trạm | — |

### TIP 4b. Mô hình năng lượng theo M300 (chỉ v2)

| Tham số | Giá trị | Nguồn | Phần dùng | Trạng thái |
|---|---|---|---|---|
| `energy_models.uav_weight` | 88.26 N (9 kg) | DJI, trang hỗ trợ Matrice 300 RTK (specs) | Max Takeoff Weight 9 kg; tải tối đa 2.7 kg (Heliguy). Thiết kế: trạm phát dùng hết tải | Trang hỗ trợ |
| `energy_models.rotor_disc_area` | 0.894 m² | DJI, specs M300 (cánh quạt 2110) | 4 cánh 21 inch; đường kính suy từ tên 2110, không thấy ghi trực tiếp | Suy luận |
| `induced_power`, `mean_induced_velocity` | 616.35 W, 6.35 m/s | Zeng, Xu & Zhang, "Energy Minimization for Wireless Communication With Rotary-Wing UAV", IEEE TWC 18(4), 2019 (arXiv:1804.02238), Eq. 61 | Pi = (1 + k) W^1.5 / sqrt(2 rho A), v0 = sqrt(W / (2 rho A)), k = 0.1 (Table I) | Như v1.1 |
| `blade_profile_power`, `rotor_tip_speed` | 444.30 W, 175.6 m/s | Hiệu chỉnh: treo hết pin 548 Wh trong 31 phút ở 9 kg (Heliguy, "DJI M300 RTK v DJI M200 Series"; bảng của Coptrz). Tốc độ đầu cánh suy từ P0 theo Eq. 61 | P0 + Pi = 1060.65 W. Coi 31 phút là thời gian treo | Nguồn bên thứ ba, không phải trang DJI. DJI ghi 55 phút không tải, 43 phút với H20T; ba mốc không khớp cùng lúc với mô hình treo của Zeng, nên chỉ khớp ở 9 kg |

Kết quả: treo cộng radio tối đa 1189 W, hết pin trong 27.7 phút (Table I của Zeng: 1500 W, 21.9 phút). Bay ngang 17 m/s: 725 W; công suất nhỏ nhất ở khoảng 19 m/s, trên giới hạn tốc độ. v1.1 (`ma3.yaml`) giữ Table I.

## TIP 7. Prior theo nhu cầu

| Tham số | Giá trị | Nguồn | Phần dùng | Trạng thái |
|---|---|---|---|---|
| `prior_mode` | `demand` | Thiết kế | Đỉnh của prior = k-means có trọng số của nhu cầu dự kiến trong episode, k = số vị trí đang phục vụ | — |
| Thuật toán | Lloyd có trọng số, khởi tạo k-means++ có trọng số, tốt nhất trong 8 lần | D. Arthur, S. Vassilvitskii, "k-means++: The Advantages of Careful Seeding", Proc. ACM-SIAM SODA 2007, pp. 1027–1035 | Khởi tạo D² | — |
| Trọng số nhu cầu | Mỗi slice có khối lượng bằng `slice_weights` | Thiết kế, theo cách tính mục tiêu | QoS và SLA lấy trung bình trong slice rồi qua các slice, nên mỗi slice nặng như nhau, không theo số UE. eMBB rải theo bản đồ mật độ của episode (Lee et al. 2014, `traffic_map`), mMTC đều trên vùng, RSU ở đúng vị trí; ô 40 m | — |

## Vùng phục vụ ngẫu nhiên theo episode và dân số dừng (2026-11-14)

| Tham số | Giá trị | Nguồn | Phần dùng | Trạng thái |
|---|---|---|---|---|
| Kích thước vùng | 2 km × 2 km (giữ) | 3GPP TR 38.901, Table 7.2-1 | So mật độ: UMa ISD 500 m (khoảng 0.22 km² mỗi trạm), UMi 200 m; 10 UAV trên 4 km² là 0.4 km² mỗi UAV (ISD tương đương khoảng 680 m), vẫn thưa hơn UMa | Như v1.1 |
| `city.random_offset` | Lưới nhà dời đều trong một bước lưới | Thiết kế | Bố cục phố khác nhau theo episode; vẫn là lưới ITU-R P.1410 | — |
| `rsu.random_offset` | Gốc lưới đường TR 36.885 đều trong một khối | Thiết kế | RSU ở chỗ khác nhau theo episode, vẫn cách 1 km theo đường và ở ngã tư | — |
| `return_home.random_edge` | Home Point và gateway đều trên mép vùng, ở ngã tư gần nhất | Thiết kế | Đội cứu hộ dựng trạm mặt đất ở chỗ tiện, không cố định | — |
| `ue_dynamics.stationary` | Poisson(91) UE đến lúc reset; số đến mỗi bước Poisson(91 × xác suất rời) | Hàng đợi M/M/∞: dân số dừng là Poisson với trung bình λ/μ (vd. L. Kleinrock, Queueing Systems Vol. 1, Wiley 1975, Sec. 3.4) | Đếm ở cuối bước, các UE đến trong bước còn ở lại tới cuối bước là Poisson(λ (1 − e^{−μT}) / μ) = 91 × xác suất rời; trung bình dừng đúng 91. v1.1: 91 lúc đầu, trôi lên khoảng 100 | Sửa hạn chế 12 |
| `traffic_patterns.embb.peak_rate` | 2 Mbps (bỏ nhãn tạm) | Ericsson, "5G NR for public safety", arXiv:2103.02434: video nhiệm vụ 150 kbps – 5 Mbps tuỳ chất lượng; NIST PSCR, "Evaluating multicast capability in LTE public safety networks": "Standard Video 1.5 Mb/s" | Trong khoảng của các nguồn | Đọc qua kết quả tìm kiếm, chưa đọc bản đầy đủ. Video nhiệm vụ thật chủ yếu là đường lên; env mô phỏng đường xuống |
