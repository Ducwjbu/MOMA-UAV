# Kế hoạch: trình xem UAV và UE theo thời gian

Thiết kế chi tiết ở [architecture.md](architecture.md). Plan này chia việc thành các bước, mỗi bước có điều kiện xong rõ ràng.

**Mục tiêu:** xem lại một episode đánh giá của các policy 3 UAV: UAV bay ra sao, UE đi ra sao, UE nào được phục vụ và trượt SLA ở đâu. Có thể đặt hai method cạnh nhau trên cùng một episode.

**Không làm:**
- không sửa env, agent hay prior;
- không đổi kết quả v1.1;
- không chạy trong lúc train;
- không đưa vào bundle Kaggle.

## Trạng thái (2026-11-13)

P0, P1, P2 xong; P3 đang làm. Kết quả P0 (script kiểm ở scratchpad, 3 góc ω × 3 episode seed 100000+, `ma3_prior` và `ma3` seed 0, checkpoint 1M):

- P0.1: đủ thuộc tính; `ue.indoor` và `ue.fixed` đọc bằng `getattr`. UE có demand nhưng không có RB: các hàm thoả mãn trả 0 mà không ghi vào UE, nên bộ ghi tự gán 0.
- P0.2: tỉ lệ đạt SLA theo slice tính từ trạng thái từng UE khớp `info['sla_*']` ở mọi bước.
- P0.3: cùng env seed, vị trí UE giống hệt nhau giữa `ma3` và `ma3_prior` ở mọi bước, nên chế độ so sánh cạnh nhau có nghĩa.
- P0.4: reward vector có và không có bộ ghi giống nhau bit-for-bit.
- P0.5: action đọc từ closure khớp `clip(residual + move_to(...))`.

## P0. Kiểm giả định trước khi viết code (khoảng 2 giờ)

| # | Kiểm tra | Cách làm | Nếu sai |
|---|---|---|---|
| P0.1 | Các thuộc tính bộ ghi cần có mặt và có nghĩa đúng: `ue.indoor` (chỉ có khi có city), `ue.fixed` (chỉ RSU có, dùng `getattr`), `ue.demand_bps`, `ue.*_satisfaction`, `uav.current_battery`, `env.wind_ref`, `env.prior_vertices` | Chạy 1 episode `ma3_prior`, in thử `snapshot(env)` | Sửa `snapshot`; ghi lại trong architecture.md |
| P0.2 | `*_satisfaction` trên UE là giá trị của bước hiện tại, không phải của bước trước, và UE không có demand thì bị bỏ qua | Đọc `_calculate_qos_reward`, so `sla` từng UE với `info['sla_*']` của bước đó | Tự tính trạng thái SLA từ các trường thô, hoặc chỉ hiện SLA theo slice |
| P0.3 | Với cùng episode seed, đường đi của UE có giống nhau giữa hai method không (RNG có bị rút theo cách phụ thuộc hành động không) | Ghi `ma3` và `ma3_prior` seed 200000, so vị trí UE từng bước | Chế độ so sánh vẫn dùng được, nhưng phải ghi rõ "UE khác nhau giữa hai bên sau bước X" |
| P0.4 | Bộ ghi không đổi kết quả | Reward vector của `eval_omega` có và không có bộ ghi, 3 episode × 3 góc ω, phải giống nhau bit-for-bit | Tìm chỗ bộ ghi gọi hàm có RNG hoặc đổi trạng thái; không làm tiếp khi chưa sửa xong |
| P0.5 | Tính lại prior trong bộ ghi khớp với prior mà env thật sự nhận | So `clip(residual + prior)` tính lại với action mà `inner_step` nhận (thử bằng cách bọc tạm trong script kiểm tra) | Đọc action đã cộng prior bằng cách khác, ví dụ bọc trước khi trainer gắn prior |

## P1. Bộ ghi `visualize/record.py` (khoảng 1 ngày)

- `snapshot(env)`: mọi chỗ đọc env gom vào đây.
- `record_episode(trainer, omega, seed, steps)`: bọc reset/step, gán track id cho UE (id tái dùng → track mới), tính lại prior và lệnh dịch chuyển.
- `static_info(env)`: lưới nhà, mật độ traffic (lượng tử 8 bit), gateway, đỉnh prior, gió.
- CLI:

  ```powershell
  py -m visualize.record --methods ma3_prior ma3 --seed_base 200000 --episodes 0 1 `
     --corners qos energy fair [--roles] [--seeds 0] [--step 1000000] [--checkpoint_glob ...] --out visualize/out/run1 [--html]
  ```

  - Mặc định lấy checkpoint 1M của seed 0 theo layout `runs/` giống `eval_trajectories.py`.
  - `--roles` thêm các góc `u{i}_energy` (ω riêng mỗi UAV).
  - `--html` ghi thêm file HTML tự chứa.
- Thêm `visualize/__init__.py`; thêm `visualize/out/` vào `.gitignore`.

**Xong khi:**
- P0.4 vẫn đạt khi bật đầy đủ bộ ghi;
- một trace cho `ma3_prior` × 3 góc × 2 episode dưới 5 MB;
- vị trí UAV trong trace khớp với `eval_trajectories.py` cùng seed.

## P2. Trình xem `visualize/viewer.html` (khoảng 1–1,5 ngày)

Làm theo thứ tự, mỗi mục dùng được ngay:

1. Bản đồ tĩnh: nhà, gateway, đỉnh prior. Vẽ UAV và UE ở khung bước; thanh thời gian bước tới/lui.
2. Play/pause, nội suy giữa hai khung bước. Khung nội suy có nhãn rõ và association bị làm mờ.
3. Màu UE theo slice hoặc theo trạng thái SLA; đường association; vệt bay của UAV; vòng power; mũi tên residual và prior; gió.
4. Rê chuột vào UE và UAV để xem chi tiết; panel bên với chuỗi thời gian và vạch bước hiện tại.
5. So sánh hai method cạnh nhau, đồng bộ thời gian.
6. Chế độ tối, bố cục hẹp. Nếu trace không nhúng sẵn thì cho chọn file JSON.

**Xong khi:**
- mở `visualize/out/run1.html` trực tiếp từ đĩa (không server, không mạng) và phát mượt với 3 UAV và khoảng 120 UE;
- đọc được trên trình xem các hành vi đã biết: `ma3_prior` bay đi bay lại quanh đỉnh ở góc QoS, `ma3` seed 3 đứng ở mép vùng.

## P3. Tài liệu và commit (khoảng 1 giờ)

- Thêm lệnh chạy vào `README.md` và mục Commands của `CLAUDE.md`, giữ hai file đồng bộ.
- Commit lên `duc` sau khi P0.4 đạt và trình xem chạy được; không commit `visualize/out/`.

## Rủi ro

| Rủi ro | Ảnh hưởng | Cách xử lý |
|---|---|---|
| Bộ ghi vô tình đổi trạng thái env hoặc RNG | Trình xem hiện một episode khác với số liệu báo cáo | Kiểm bit-for-bit P0.4, chạy lại sau mỗi lần sửa `snapshot` |
| Đọc nhầm khung nội suy là trạng thái thật | Kết luận sai về coverage trong lúc bay | Nhãn rõ, association mờ, mặc định có thể đặt 0 khung nội suy |
| Env đổi tên thuộc tính (env v2) | Bộ ghi gãy | Mọi chỗ đọc ở `snapshot`; trace ghi `git_commit` |
| Vòng kín hỗn loạn: GPU và CPU cho quỹ đạo khác nhau | Trình xem không khớp số liệu chạy trên Kaggle | Luôn ghi trên CPU, giống `eval_checkpoint.py`; ghi `device` vào trace |
| UE khác nhau giữa hai method cùng seed (P0.3) | So sánh cạnh nhau kém ý nghĩa | Phát hiện và báo bước lệch đầu tiên trong trình xem |

## Sau đó (chưa lên lịch)

- **Chẩn đoán QoS ở trung điểm đường bay,** để kiểm giả thuyết policy bay đi bay lại vì mạng chỉ được tính ở điểm cuối. Đây là phép đo mới, cần chạy lại phần mạng của env trên bản sao và lưu/khôi phục RNG (xem architecture.md, mục 6).
- **Đưa heuristic và đội hình không train vào trình xem.**
- **Ghi λ và ω theo thời gian từ `eval_controller.py`.**
