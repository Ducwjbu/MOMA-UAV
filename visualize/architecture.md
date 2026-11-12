# Kiến trúc: trình xem UAV và UE theo thời gian

Mục đích: xem lại một episode đánh giá, gồm các UAV bay ra sao và các UE di chuyển ra sao, để debug hành vi của policy. Công cụ này chỉ dùng để quan sát: không sửa env và không đổi bất kỳ kết quả nào.

## 1. Tổng quan

```
checkpoint (.pth) + train config ──► record.py ──► trace JSON ──► viewer.html (+ JSON nhúng sẵn) ──► trình duyệt
                                         │
                       bọc env.reset / env.step của trainer.eval_env,
                       chỉ đọc trạng thái sau mỗi bước
```

Gồm hai phần:

| Phần | File | Vai trò |
|---|---|---|
| Bộ ghi | `visualize/record.py` | Nạp checkpoint giống `eval_checkpoint.py`, chạy `agent.eval_omega` trên episode seed `seed_base + k`, ghi trạng thái sau `reset` và sau mỗi `step` |
| Trình xem | `visualize/viewer.html` | Một file HTML tĩnh, JS thuần và canvas, không thư viện ngoài, mở offline được. `record.py --html` chép template và nhúng trace vào để ra một file tự chứa |

Output ghi vào `visualize/out/`, thư mục này thêm vào `.gitignore`.

## 2. Những gì env thực sự tính (quyết định cách hiển thị)

- Một bước env là `T_L` = 60 s. Mỗi UAV nhận một hành động cho cả 60 s.
- Năng lượng propulsion coi UAV bay thẳng với vận tốc không đổi từ vị trí cũ đến vị trí mới. Vị trí mới là vị trí sau mọi giới hạn: airspeed, clip vùng bay, đẩy tách, độ cao trên mái nhà ([environment.py:844-908](../environment.py#L844-L908)).
- Mạng chỉ được tính một lần mỗi bước, tại vị trí cuối của UAV. Đó là UE di chuyển, LoS và shadowing, association, RB, SINR, rate, queue và reward ([environment.py:917-956](../environment.py#L917-L956)). Trong lúc UAV đang bay, không có gì về mạng được tính.
- UE di chuyển một lần mỗi bước. UE ngoài trời đi khoảng 50 m dọc phố; UE indoor và RSU đứng yên. UE mới đến xuất hiện, UE rời đi thì `is_active = False` rồi bị xoá.

Vì vậy trình xem có hai loại khung hình:

- **Khung bước (keyframe):** trạng thái thật sau `reset` (t = -1) và sau mỗi `step` (t = 0..17). Association, màu SLA và số liệu chỉ đúng ở các khung này.
- **Khung nội suy:** UAV và UE đi thẳng giữa hai khung bước. Phần chuyển động này khớp với giả định năng lượng. Trong lúc nội suy, đường association và màu SLA bị làm mờ, và có nhãn "nội suy, mạng không được tính" để không ai đọc nhầm.

## 3. Bộ ghi (`record.py`)

### 3.1 Cách móc vào

- Dùng lại `eval_checkpoint.load_trainer(config, checkpoint)`. Prior đội hình được gắn qua trainer, nên `trainer.eval_env.step` đã là bản có prior.
- Bọc `env.reset` và `env.step` ở lớp ngoài cùng, giống `eval_trajectories.record_episodes`, rồi gọi `trainer.agent.eval_omega(env, omega, 1, eval_steps)` với `agents_moma_ac.EVAL_SEED = seed_base + k`. Episode trong trình xem vì thế chính là episode k của `eval_checkpoint.py` cùng `--seed_base`.
- Lớp ngoài cùng chỉ thấy **residual**, tức output của actor. Action mà env thật sự nhận (residual + prior, đã clip) được đọc trực tiếp: bộ ghi tạm thay `inner_step` trong closure của `prior.attach` bằng một hàm ghi lại action rồi gọi `inner_step` gốc, và trả lại như cũ khi xong. Cách này chính xác hơn tính lại prior, và không phụ thuộc cách prior chọn đỉnh khi có `fleet`.
- Bộ ghi chỉ đọc thuộc tính, không gọi hàm nào của env có dùng RNG, và không đổi thứ tự gọi. Điều kiện bắt buộc: reward vector có và không có bộ ghi phải giống nhau bit-for-bit (plan, P0.4, đã đạt).

### 3.2 Chuỗi hành động của mỗi UAV

Ghi đủ các khâu để thấy khâu nào cắt hành động:

```
residual (actor) ─► + prior move_to ─► clip [-1,1] ─► × velocity_max·T_L ─► giới hạn airspeed (gió)
   ─► clip vùng bay ─► đẩy tách ─► độ cao trên mái ─► vị trí cuối
```

- **Ghi trực tiếp:** residual, action env nhận (sau prior và clip), vị trí trước bước, vị trí sau bước.
- **Tính lại:** điểm đến của lệnh (`cmd_target`, các dòng đầu của `NetworkSlicingEnv.step` trước giới hạn airspeed), tốc độ mặt đất.
- **Chênh lệch giữa độ dời được lệnh và độ dời thật** là phần do env cắt hoặc đẩy. Bộ ghi không phân tách từng khâu bên trong env, để khỏi phải sửa env.

### 3.3 Định danh UE

Env tái dùng id của UE đã rời đi (`ue_id_pool`, [environment.py:1500](../environment.py#L1500)). Do đó id không đủ để nối một UE qua các khung. Bộ ghi gán **track id** = (id, bước xuất hiện). Một id biến mất rồi xuất hiện lại là hai track khác nhau, và trình xem không nội suy giữa chúng.

### 3.4 Lược đồ trace (JSON)

```jsonc
{
  "meta": {
    "method": "ma3_prior", "checkpoint": "...", "config": "...", "env_config": "...",
    "seed_base": 200000, "episode": 0, "omega": [[1,0,0],[1,0,0],[1,0,0]],   // một hàng mỗi UAV
    "T_L": 60, "git_commit": "5db44e6", "device": "cpu"
  },
  "static": {                                   // không đổi trong episode
    "area": [2000, 2000], "fly_range": {"x": [..], "y": [..], "h": [..]},
    "city": {"pitch": 57.7, "width": 40.8, "offset": 8.45, "heights": [[...]]},   // lưới nx × ny
    "traffic_density": {"pixel": 10, "shape": [200, 200], "data_b64_u8": "..."},   // log, lượng tử 8 bit
    "gateway": [x, y, h], "prior_vertices": [[x, y, h], ...] | null,
    "wind": [-3.37, 5.09],                       // m/s ở độ cao tham chiếu (x đông, y bắc), hướng gió thổi tới
    "v_max": 17                                  // một action di chuyển bằng 1 là v_max * T_L
  },
  "frames": [                                   // t = -1 (sau reset), 0..T-1
    {
      "t": -1,
      "uavs": [{"id": 0, "active": true, "pos": [x,y,h], "power": 0.71, "battery_wh": 520,
                "residual": [dx,dy,dh] | null, "prior": [..] | null, "commanded_disp": [..] | null,
                "bw": {"embb": .., "urllc": .., "mmtc": ..} | null, "n_served": 31}],
      "ues": [{"track": 17, "id": 4, "pos": [x,y], "slice": "embb", "indoor": true, "rsu": false,
               "uav": 1 | null, "status": 4, "thr": 0.009, "delay": 0.17, "rel": 1.0,
               "rate_mbps": 0.47, "demand_bps": 5e7, "delay_ms": 27.5}],   // thực tế lưu theo cột, mỗi trường một mảng
      "team": {"reward": [qos, -energy, fair], "coverage": .., "handovers": .., "sla": {"embb": ..}}
    }
  ]
}
```

- **Mã trạng thái UE:** 0 không có traffic, 1 đạt SLA, 2 không được phủ, 3 có UAV nhưng không có RB, 4/5/6 trượt SLA vì rate/delay/reliability (khâu có mức thoả mãn thấp nhất). Cùng quy tắc với `_calculate_qos_reward`: tỉ lệ đạt theo slice khớp `info['sla_*']` từng bước (P0.2).
- Trace thật gom các episode vào `{"meta", "scenes", "episodes"}`. Cảnh (`scenes`, city và traffic map) được khử trùng lặp theo hash, vì cùng env seed cho cùng cảnh ở mọi method và góc ω.
- **Kích thước:** khoảng 120 UE × 19 khung, cộng một lưới nhà khoảng 35 × 35. Thực đo: 2,1 MB cho 12 episode (2 method × 3 góc × 2 episode), nên nhúng thẳng vào HTML.
- Có thể ghi nhiều episode và nhiều method vào cùng một file; trình xem cho chọn.

## 4. Trình xem (`viewer.html`)

### 4.1 Lớp vẽ (bật/tắt từng lớp)

1. **Nền:** footprint toà nhà, xám đậm nhạt theo chiều cao. Heatmap mật độ traffic, tắt mặc định.
2. **Gateway/Home Point.** Các đỉnh prior, vẽ nét đứt.
3. **UE:** chấm tô màu theo **slice** hoặc theo **trạng thái SLA** (đạt / trượt / không phủ / không có traffic).
   - Viền dày cho UE indoor; ký hiệu vuông cho RSU.
   - UE mới đến hiện dần, UE rời đi mờ dần.
   - Vệt đường đi của UE, tắt mặc định.
4. **Đường association** từ UE đến UAV phục vụ, cùng màu với UAV.
5. **UAV:**
   - biểu tượng kèm nhãn độ cao;
   - vòng tròn có bán kính tỉ lệ với power;
   - vệt đường bay N bước gần nhất;
   - mũi tên residual và mũi tên prior ở khung bước.
6. **Mũi tên gió** ở góc bản đồ.

### 4.2 Điều khiển

- Play/pause, chọn tốc độ, kéo thanh thời gian, bước tới/lui từng khung bước. Ô chọn số khung nội suy giữa hai bước (0 = chỉ khung bước).
- Chọn method, episode và góc ω.
- **So sánh:** hai bản đồ cạnh nhau, đồng bộ thời gian, cùng episode seed, hai method khác nhau.
- **Rê chuột:**
  - vào UE: hiện track, slice, indoor, UAV phục vụ, ba mức thoả mãn, khâu làm trượt SLA;
  - vào UAV: hiện cả chuỗi hành động ở 3.2.

### 4.3 Panel bên

- Theo thời gian, có vạch ở bước hiện tại: reward vector, coverage, SLA theo slice, power và tốc độ từng UAV.
- Bảng ở bước hiện tại: số UE phục vụ, bandwidth eMBB/URLLC/mMTC và pin của mỗi UAV.

### 4.4 Kỹ thuật

- JS thuần với Canvas 2D: khoảng 1200 hình chữ nhật nhà vẽ một lần vào canvas nền, UE và UAV vẽ lại mỗi khung.
- Không CDN, mở trực tiếp từ đĩa.
- Màu theo token CSS, có chế độ tối.
- Template đọc trace từ thẻ `<script type="application/json" id="trace">`. Nếu thẻ rỗng, template cho chọn file JSON từ đĩa, để dùng chung một template cho mọi trace.

## 5. Ràng buộc và quan hệ với phần còn lại

- **Không sửa** `environment.py`, `agents_moma_ac.py`, `prior.py`, nên env v1.1 không đổi. Không cần đưa vào bundle Kaggle; chạy local trên CPU (`MOMA_DEVICE=cpu`), giống `eval_checkpoint.py`.
- Chạy từ gốc repo, vì `environment.py` đọc config theo đường dẫn tương đối: `py -m visualize.record ...` (cần `visualize/__init__.py`).
- Trọng số: mặc định 3 góc như `eval_trajectories.py`; `--roles` thêm các góc ω riêng mỗi UAV, `--omega Q,E,F` thêm trọng số bất kỳ, `--sweep N` thêm các trọng số bên trong của sweep eval lúc train (`eval_pareto_front`). Trình xem chỉ phát lại episode đã ghi: muốn xem một ω thì phải ghi nó trước, vì trang không chạy policy.
- Dùng lại `eval_checkpoint.load_trainer`, `prior.formation_vertices` / `move_to` / `assign_vertices`, và `eval_trajectories.METHODS` để ánh xạ tên method sang config.
- **Bộ ghi phụ thuộc tên thuộc tính của env.** Bộ ghi đọc các thuộc tính sau:
  - UAV: `uavs[i].position`, `current_power`, `current_battery`, `active`;
  - UE: `ues[*].position`, `slice_type`, `assigned_uav`, `is_active`, `indoor`, `fixed`, `demand_bps`, `*_satisfaction`;
  - bản đồ và đội hình: `city`, `traffic_map`, `backhaul_gateway`, `prior_vertices`;
  - gió: `wind_ref`.

  Mọi chỗ đọc gom vào một hàm `snapshot(env)`, để khi env đổi tên chỉ phải sửa một chỗ.

## 6. Ngoài phạm vi (có thể làm sau)

- **Chẩn đoán "mạng giữa đường bay":** tính QoS khi đặt UAV ở trung điểm đường bay, chỉ để đo. Việc này phải chạy lại phần mạng của env trên một bản sao, đồng thời lưu và khôi phục `np.random`, vì chính phần mạng rút số ngẫu nhiên (shadowing, `rb_order`). Đây là phép đo mới chứ không còn là hiển thị, nên để thành một bước riêng sau khi trình xem chạy được.
- **Chạy env trực tiếp từng bước (live):** cần server.
- **Xem chi tiết từng RB.**
- **Heuristic và đội hình không train** (`eval_heuristics.py`, `eval_formation.py`): đưa vào sau, qua cùng hàm `snapshot`.
