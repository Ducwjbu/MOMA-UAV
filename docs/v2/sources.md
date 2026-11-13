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
