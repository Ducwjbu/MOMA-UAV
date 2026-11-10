# Bài toán và phương pháp

## Bài toán

Ba UAV làm trạm phát sóng (UAV base station) phục vụ một khu đô thị dày đặc 2 km × 2 km, chia mạng thành 3 slice:

| Slice | Người dùng | Yêu cầu (SLA) | Nguồn |
|---|---|---|---|
| eMBB | 31 UE di chuyển (đi bộ 3 km/h theo đường phố), có tải với xác suất 0.2 mỗi bước | 50 Mbps, 10 ms, độ tin cậy 0.99 | TS 22.261 Table 7.1-1 (tốc độ); độ trễ và độ tin cậy là giả định thiết kế |
| URLLC | 26 thiết bị bên đường (RSU) cố định, mỗi 1 km trên lưới đường | 10 Mbps, 30 ms, 0.99999 | TS 22.261 Table 7.2.3.2-1 |
| mMTC | 60 cảm biến, gửi báo cáo định kỳ rất nhỏ | được phục vụ, 10 s, 0.95 | TR 45.820; TR 38.913 Sec. 7.6 |

Mỗi bước mô phỏng là 60 giây; một episode dài 18 bước (18 phút). Đây là nhiệm vụ dài nhất mà pin vẫn còn đủ để bay về điểm xuất phát trong trường hợp xấu nhất về gió.

Mỗi UAV chọn 13 giá trị mỗi bước: di chuyển (x, y, độ cao), công suất phát, và chia băng thông cho 9 vùng nhu cầu (3 slice × gần/vừa/xa).

### Ba mục tiêu (đều nằm trong [0, 1])

1. **QoS:** mức hài lòng trung bình của từng slice, rồi lấy trung bình 3 slice với trọng số bằng nhau. Mỗi UE lấy chỉ số tệ nhất trong ba chỉ số tốc độ, độ trễ, độ tin cậy; giá trị này bằng 1 khi đạt đủ SLA.
2. **Năng lượng** (lấy dấu âm): công suất tiêu thụ trung bình của bay và phát sóng, chuẩn hoá min–max. Mô hình bay theo Zeng et al. 2019: lơ lửng (hover) tốn khoảng 1.37 kW, bay ở tốc độ tối ưu chỉ khoảng 0.94 kW.
3. **Công bằng:** chỉ số Jain trên tỉ lệ được phục vụ `min(thông lượng / tải, 1)` của các UE có tải (Jain, Chiu & Hawe 1984).

Trọng số ω trên 3 mục tiêu do người vận hành chọn lúc chạy. Một policy duy nhất phải tốt ở mọi ω, nên kết quả là cả một front Pareto.

### Môi trường thực tế hoá (v1.1)

Mỗi thành phần đều có nguồn; chi tiết nằm trong comment của `config/environment/ma3.yaml` và trong CLAUDE.md.

- **Thành phố:** lưới nhà ITU-R P.1410, tính tầm nhìn thẳng (LoS) theo hình học. UE trong nhà chịu suy hao xuyên tường theo TR 38.901, cộng shadowing tương quan theo khoảng cách.
- **Kênh truyền:**
  - fading Rician/Rayleigh;
  - anten 3GPP TR 38.901;
  - tốc độ theo Shannon có suy giảm (Mogensen et al.);
  - băng tần NR n78: 3.5 GHz, 273 RB × 360 kHz.
- **Gió Hà Nội:**
  - tốc độ theo phân bố Weibull từ Global Wind Atlas;
  - hướng theo hoa gió trạm Láng;
  - mô hình năng lượng có gió của Dai et al.
- **Backhaul mmWave:** Galkin et al.
- **UAV:** pin 548 Wh (2 pin DJI TB60), tối đa 17 m/s, độ cao 50–120 m, tự quay về khi pin yếu (theo M300 RTH).
- **Phân bố người dùng eMBB:** bản đồ mật độ log-normal tương quan không gian của Lee et al. 2014.

## Các phương pháp so sánh

| Tên trong repo | Vai trò | Mô tả |
|---|---|---|
| `ma3` | Baseline (paper) | MOMA-TD3, Alg. 1 của paper MOMA-AC: TD3 có điều kiện theo ω, CTDE (critic tập trung, actor phân tán), γ 0.99 như paper |
| họ heuristic | Baseline không học | 10 cách bay tam giác r = 600 m quanh tâm khu vực: đứng yên, xoay đều, dao động; công suất 1 hoặc 0.05; chia băng thông đều hoặc ưu tiên URLLC. Front của họ là tập 10 điểm, ở mỗi ω lấy cấu hình tốt nhất |
| tam giác đứng yên | Tham chiếu | 3 UAV đứng yên ở tam giác, công suất 0.5, không tốn chi phí bay ra |
| `ma3_prior` | **Đề xuất** | Residual policy trên đội hình tam giác, γ 0.9 |
| `ma3_g90` | Ablation | `ma3` với γ 0.9 (không có prior) |
| `ma3_prior_g99` | Ablation | `ma3_prior` với γ 0.99 |

## Phương pháp đề xuất: residual policy trên đội hình

**Ý tưởng:** chuyển động của UAV i bằng "bay tới đỉnh i của tam giác" cộng một phần hiệu chỉnh do mạng nơ-ron học:

```
di chuyển_i = clip(move_to(đỉnh_i) + a_i[0:3], -1, 1)
```

Đây là residual policy learning (Silver et al. 2018; Johannink et al. 2019, Eq. 5).

- **Khởi tạo bằng heuristic:** các hàng di chuyển ở lớp cuối của actor được khởi tạo bằng 0, nên lúc chưa học policy chính là heuristic tam giác.
- **Phần còn lại do mạng tự học:** công suất và băng thông không có prior.
- **Tam giác r = 600 m:** lệch chưa tới 0.4% so với vị trí phủ tối ưu theo 3-means của khu 2 km.

**Vì sao cần prior?** Kiểm tra bằng học có giám sát cho thấy vị trí UAV chỉ giải thích khoảng 17% phương sai QoS của một bước. Tín hiệu quá yếu, nên MOMA-TD3 học từ đầu không tìm được đội hình dàn trải.

**Vì sao γ 0.9?**
- **Về lý thuyết:** với γ 0.99, critic dạng vector trôi xuống thấp hơn giá trị thật rất nhiều. Ở góc năng lượng nó thậm chí âm cho QoS và công bằng, đúng hiện tượng mà TD3 paper mô tả ở Sec. 5.1. Sự trôi này kéo phần hiệu chỉnh ra xa đội hình.
- **Về tầm nhìn:** tác động của một hành động tắt dần sau khoảng 2 bước, trong khi episode dài 18 bước.
- **Cách chọn:** γ 0.9 được chọn trước từ lập luận này và một pilot thô, không phải từ một đợt tuning có ngân sách như nhau cho mọi phương pháp. Phải nói rõ điều này trong báo cáo.

## Mạng và huấn luyện

- **Actor:** một thân chung, mỗi UAV một đầu ra riêng. Actor của UAV i chỉ thấy quan sát của nó (41 chiều) và ω.
- **Critic:** thấy quan sát và hành động chung của cả đội. Mỗi mục tiêu có một Q riêng, nên critic ra vector Q.
- **Hàm mất mát của actor:** vô hướng hoá bằng `Q · ω`.
- **Chọn ω:** mỗi episode rút ω ngẫu nhiên từ Dirichlet(1, 1, 1).
- **Số bước:** 1 triệu bước mỗi run. Trên Kaggle T4, chạy 2 run song song, mỗi run mất khoảng 9.5 giờ.
