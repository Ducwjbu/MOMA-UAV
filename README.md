# MOMA-UAV

Học tăng cường đa mục tiêu điều khiển UAV làm trạm gốc, cắt lát mạng (network slicing) cho ba slice eMBB, URLLC và mMTC. Agent là TD3 có điều kiện theo vector trọng số omega. Agent học một Pareto front trên ba mục tiêu: QoS, năng lượng và fairness.

Repo có hai bài toán:

- **1 UAV**: so sánh các cách chọn omega khi train. `random` là baseline (MOMA-AC gốc, lấy mẫu Dirichlet đều), so với `ols`, `gpi_ls` và `gpi_pd`.
- **3 UAV (`ma3`)**: MOMA-TD3 của paper MOMA-AC, huấn luyện tập trung, thực thi phân tán (CTDE). Hiện chỉ có lấy mẫu omega đều. Bản cải tiến `ma3_prior` cho actor học phần lệch (residual) trên một đội hình nền cố định, với γ = 0,9; `ma3` giữ làm baseline.

Mô tả kỹ thuật đầy đủ (môi trường, reward, agent, các lỗi đã biết) nằm trong [CLAUDE.md](CLAUDE.md). Hướng dẫn chạy trên Kaggle nằm trong [kaggle/README.md](kaggle/README.md).

## Phiên bản môi trường

Môi trường `v1.0` (git tag `v1.0`) có thành phố tường minh theo ITU-R P.1410, suy hao xuyên tường, shadowing tương quan, fast fading, gió Hà Nội, traffic theo chuẩn 3GPP, các trạm bên đường (RSU) cho URLLC, và QoS theo SLA. Nguồn của từng tham số được ghi trong comment của `config/environment/*.yaml`.

Sau `v1.0` môi trường đổi hai lần, tạo thành `v1.1` (git tag `v1.1`). Kết quả của báo cáo môn học chạy trên `v1.1`:

- **Pin dự trữ để bay về:** theo cơ chế Low Battery RTH của DJI M300. Mỗi episode dài 18 bước với 3 UAV, 19 bước với 1 UAV.
- **Mô hình lưu lượng mới:** bản đồ mật độ lưu lượng log-normal của Lee et al. 2014 (`traffic_map.py`, mục `traffic_map` trong config) thay cho các hotspot. Mỗi episode sinh một bản đồ mới. UE eMBB xuất hiện theo bản đồ, UE mMTC xuất hiện đều.


**Mọi thay đổi môi trường đều làm kết quả cũ mất giá trị so sánh**, vì thang reward thay đổi. Vì vậy:

- Chỉ so những run cùng phiên bản môi trường.
- `results-v1.1/` là kết quả của môi trường `v1.1`: CSV lúc train của 4 biến thể 3 UAV × 5 seed, và eval trên seed test (`eval_test/`, bảng gộp `agg.csv`).
- Kết quả của các môi trường cũ đã được xoá khỏi thư mục, vẫn còn trong lịch sử git (tag `v1.0`).

Muốn sửa môi trường thì bàn trước với cả nhóm. Nếu sửa, tạo tag mới (`v1.1`, ...) và chạy lại toàn bộ.

**Môi trường v2 (khoá luận, đang xây):** `config/environment/v2.yaml`, kế hoạch trong `docs/v2/plan.md`. Mỗi tính năng v2 là một khoá config, không có khoá thì code chạy đúng như `v1.1`. Sau mỗi thay đổi code, chạy `py docs/v2/check_v11.py check` để chắc `ma3.yaml` và `default.yaml` vẫn cho kết quả giống `v1.1` từng bit (lần đầu chạy `record` trên code `v1.1` để có tham chiếu; tham chiếu nằm trong `runs/`, không commit).

## Cài đặt

Cần Python 3.10 trở lên (đã thử với 3.12). GPU NVIDIA không bắt buộc nhưng nhanh hơn nhiều.

```bash
git clone https://github.com/Ducwjbu/MOMA-UAV.git
cd MOMA-UAV
pip install -r requirements.txt
```

Trên Windows, nếu lệnh `python` không có trong PATH thì dùng `py` thay cho `python` ở mọi lệnh bên dưới.

`requirements.txt` cài `torch` bản mặc định của pip. Muốn dùng GPU thì cài `torch` bản CUDA theo hướng dẫn ở [pytorch.org](https://pytorch.org/get-started/locally/). Code tự dùng GPU nếu có. Biến môi trường `MOMA_DEVICE=cpu` hoặc `MOMA_DEVICE=cuda` ép chọn thiết bị.

**Luôn chạy lệnh từ thư mục gốc của repo.** Môi trường đọc `config/environment/default.yaml` theo đường dẫn tương đối ngay lúc import.

## Chạy thử (khoảng 1–2 phút)

Nên chạy thử mỗi khi vừa pull code mới hoặc vừa sửa gì đó. Tạo file `smoke.yaml` ngoài repo, hoặc trong một thư mục không commit, với nội dung:

```yaml
# Chép nguyên file config/train/moma_ac_<variant>.yaml muốn thử, rồi sửa các dòng sau
total_training_steps: 1500
min_buffer_size: 100
batch_size: 32
max_steps_per_episode: 20
save_interval: 1500
eval_interval: 750
eval_n_omega: 5
eval_n_episodes: 1
eval_steps: 5
save_dir: "smoke_out"
```

Các khóa không ghi sẽ lấy giá trị mặc định. Với `ols`, `gpi_ls` và `gpi_pd`, thêm:

```yaml
gpi_ls_interval: 300
gpi_ls_episodes: 1
dyna_interval: 100
dyna_rollouts: 100
model_fit_steps: 30
```

Rồi chạy `python train_moma_ac.py --config smoke.yaml`. Tốc độ (it/s) trong 5000 bước đầu cao hơn hẳn lúc train thật, vì khi đó chưa cập nhật mạng; muốn đo tốc độ thì xem sau khoảng 20k bước. Chạy thử thành công khi lệnh kết thúc không lỗi và `smoke_out/` có `eval_metrics.csv`, `checkpoints/` và file `DONE`. Các số trong lần chạy thử không có ý nghĩa.

## Train

```bash
# 3 UAV: bản cải tiến (đội hình nền + residual) và baseline MOMA-TD3
python train_moma_ac.py --config config/train/moma_ac_ma3_prior.yaml --seed 1
python train_moma_ac.py --config config/train/moma_ac_ma3.yaml --seed 1
# Ablation 2x2 (đội hình nền x gamma): MOMA-TD3 với γ 0,9, và đội hình nền với γ 0,99 của paper
python train_moma_ac.py --config config/train/moma_ac_ma3_g90.yaml --seed 1
python train_moma_ac.py --config config/train/moma_ac_ma3_prior_g99.yaml --seed 1

# 1 UAV, 4 cách chọn omega
python train_moma_ac.py --config config/train/moma_ac_random.yaml --seed 1   # baseline
python train_moma_ac.py --config config/train/moma_ac_ols.yaml    --seed 1
python train_moma_ac.py --config config/train/moma_ac_gpi_ls.yaml --seed 1
python train_moma_ac.py --config config/train/moma_ac_gpi_pd.yaml --seed 1
```

Mỗi config train 1M bước. Mỗi bước là 60 giây mô phỏng. Một episode là một lượt bay: 18 bước với `ma3`, 19 bước với 1 UAV. Đó là lượt bay dài nhất mà pin vẫn còn đủ để bay về trạm mặt đất (`return_home` trong config môi trường) kể cả khi hover hết công suất ở điểm xa nhất. Episode cũng kết thúc sớm nếu pin của một UAV chỉ còn đủ để bay về.

**Đội hình nền (`formation_prior`, file `prior.py`), dùng trong `moma_ac_ma3_prior.yaml`.**
- Mỗi UAV có một đỉnh cố định của tam giác đều bán kính `prior_radius` (600 m) quanh tâm vùng, ở độ cao tối đa. Đội hình này cách vị trí phủ đều tối ưu kiểu k-means chưa tới 0,4%.
- Lệnh di chuyển gửi cho môi trường là lệnh bay về đỉnh cộng với output của actor. Actor chỉ học phần lệch đó, khởi tạo bằng 0, nên lúc đầu agent chính là đội hình (residual policy learning). Công suất và băng thông vẫn do actor quyết định.
- Config này dùng γ = 0,9, chọn trước theo tầm nhìn của bài toán. Với γ = 0,99 phần lớn giá trị mà critic phải ước lượng nằm sau mốc 18 bước, mà quan sát không cho biết còn bao nhiêu bước (xem CLAUDE.md, mục formation prior).
- Môi trường không đổi. Eval (`eval_checkpoint.py`) tự áp đội hình theo config train.

Tùy chọn thử nghiệm khác cho `ma3`: `guide_prob` và `guide_until` bật bộ dẫn đội hình (`guide.py`). Nó lái phần di chuyển trong một tỉ lệ episode để agent được thấy các đội hình dàn đều. Mặc định tắt, chỉ giúp HV khoảng +0,01, và không dùng chung được với `formation_prior`.

Các tham số:

| Tham số | Ý nghĩa |
|---|---|
| `--config` | Config train trong `config/train/`. Config tự trỏ tới config môi trường qua khóa `env_config`. |
| `--seed n` | Seed ngẫu nhiên. Kết quả ghi vào `<save_dir>/seed_<n>/`. |
| `--max_hours h` | Sau `h` giờ thì lưu trạng thái và dừng. |
| `--resume` | Chạy tiếp từ `<save_dir>/seed_<n>/resume/`. Phải dùng cùng `--config` và `--seed`. |
| `--env_config` | Thay config môi trường. Chỉ dùng cho thí nghiệm độ nhạy. |
| `--checkpoint` | Chỉ nạp trọng số. Bộ đếm bước về 0 và buffer rỗng, nên không dùng để chạy tiếp. |

**Chia seed trong nhóm.** Mỗi người nhận seed riêng để không chạy trùng, và ghi lại ai chạy seed nào. Seed 0–4 của 4 biến thể 3 UAV đã chạy xong (`results-v1.1/`).

**Chạy tiếp khi phải tắt máy.** Chạy với `--max_hours`, ví dụ `--max_hours 8`. Khi muốn tiếp tục, chạy lại đúng lệnh cũ và thêm `--resume`. Mạng, optimizer, buffer, RNG và log đều được giữ. Riêng episode đang chạy dở thì bắt đầu lại, vì trạng thái môi trường không được lưu. Trạng thái resume cũng được lưu ở mỗi `save_interval` (100k bước), nên nếu máy tắt đột ngột thì chỉ mất phần sau lần lưu cuối. Run xong sẽ xóa `resume/` và ghi file `DONE`.

**Thời gian ước tính:**

| Run | Kaggle GPU | Laptop GTX 1650 + CPU 12 luồng |
|---|---|---|
| `ma3`, 1M bước | khoảng 9 giờ (30 bước/giây, đã đo) | khoảng 14 giờ (20 bước/giây mỗi run khi chạy 3 run cùng lúc) |
| 1 UAV, 1M bước | chưa đo với `v1.0` | chưa đo với `v1.0` |

Run dài nên chạy trên Kaggle, xem [kaggle/README.md](kaggle/README.md). Nhớ build lại file upload bằng `python kaggle/build_bundle.py` mỗi khi sửa code.

## Kết quả của mỗi run

Trong `<save_dir>/seed_<n>/`:

| File | Nội dung |
|---|---|
| `eval_metrics.csv` | Mỗi `eval_interval` một dòng: `hv` (hypervolume), `eum`, `eu_front`, `n_pareto`, trung bình các mục tiêu, và tỷ lệ UE đạt đủ SLA theo slice (`avg_sla_*` qua mọi omega, `wqos_sla_*` ở omega = (1,0,0)). Với `ma3` có thêm `coverage`, `handovers`, `backhaul_limited` và khoảng cách giữa các UAV. |
| `pareto_points.csv` | Từng omega đã eval cùng reward vector và SLA, có thể tự lọc Pareto front từ đây. |
| `step_metrics.csv` | Reward của episode train gần nhất, mỗi 5000 bước. |
| `gpi_ls_metrics.csv` | Chỉ có với `ols`, `gpi_ls` và `gpi_pd`: kích thước tập omega và độ cải thiện. |
| `checkpoints/` | Trọng số mạng, mỗi 100k bước. |

Eval lúc train chỉ dùng 3 episode, lặp lại cùng seed, nên chỉ dùng để xem đường học. Số liệu cuối phải lấy theo mục dưới đây.

## Số liệu cuối cho bài báo

### 1. Eval checkpoint trên 20 episode riêng

Eval chạy trên các episode chưa từng dùng khi train (seed 100000 trở đi), trên CPU để lặp lại được, có khoảng tin cậy 95% bootstrap.

Seed 100000 trở đi đã được dùng khi phát triển `ma3_prior` (chọn γ, bán kính đội hình), nên coi chúng là tập validation. Số liệu cuối của 3 UAV dùng tập test riêng: thêm `--seed_base 200000` cho mọi policy, kể cả heuristic, và chỉ chạy khi đã chốt phương pháp.

```bash
# Baseline trước
python eval_checkpoint.py --config config/train/moma_ac_random.yaml \
    --checkpoint saved_models_moma_ac_random/seed_1/checkpoints/checkpoint_step_1000000.pth \
    --out eval_random_s1

# Variant khác, so từng cặp với baseline trên cùng episode
python eval_checkpoint.py --config config/train/moma_ac_ols.yaml \
    --checkpoint saved_models_moma_ac_ols/seed_1/checkpoints/checkpoint_step_1000000.pth \
    --baseline eval_random_s1_actor.npy --out eval_ols_s1

# 3 UAV và heuristic tam giác cố định
python eval_checkpoint.py --config config/train/moma_ac_ma3.yaml \
    --checkpoint saved_models_moma_ac_ma3/seed_1/checkpoints/checkpoint_step_1000000.pth \
    --out eval_ma3_s1
python eval_checkpoint.py --formation --out eval_triangle
python eval_heuristics.py --out eval_heuristics      # Họ 10 heuristic, cùng episode
```

Kết quả nằm trong `<out>.csv`, mỗi policy một dòng:

- `hv`, `eum`, `eu_front` và khoảng tin cậy `*_lo`, `*_hi`.
- Với `--baseline`, chênh lệch so với baseline `d_hv`, `d_eum` và khoảng tin cậy của chênh lệch.
- Reward ở 3 góc: `w_qos_*`, `w_energy_*`, `w_fair_*`.
- Tỷ lệ đạt SLA theo slice.

Các file `.npy` lưu số liệu từng episode. Với `gpi_ls` và `gpi_pd` có hai dòng: `gpi` (suy luận GPI) và `actor` (actor thường). So `actor` với baseline để tách tác dụng của cách chọn omega khỏi tác dụng của GPI.

Khoảng tin cậy ở đây chỉ phản ánh nhiễu giữa các episode eval, không phản ánh khác biệt giữa các seed train. Muốn kết luận về phương pháp thì cần nhiều seed.

### 2. Gộp nhiều seed

```bash
python compare_runs.py saved_models_moma_ac_random saved_models_moma_ac_ols saved_models_moma_ac_gpi_ls saved_models_moma_ac_gpi_pd
```

Lệnh này in mean ± std qua các seed ở mốc eval cuối, và ghi `comparison.csv` cho mọi mốc eval. Mỗi thư mục là `save_dir` của một variant, chứa các `seed_*/`. Đây là eval lúc train, chỉ để xem đường học.

Số liệu cuối gộp từ các file `.npy` của `eval_checkpoint.py` (bước 1, cùng `--seed_base`, mỗi seed một file):

```bash
python aggregate_seeds.py --method ma3 "evals/ma3_s*_actor.npy" --method ma3_prior "evals/ma3_prior_s*_actor.npy"     --baseline ma3 --out agg
```

`hv`, `eum`, `eu_front` là trung bình qua các seed. Khoảng tin cậy 95% bootstrap hai tầng (lấy lại seed, rồi lấy lại episode eval), nên phản ánh cả khác biệt giữa các seed train. `d_*_vs_<baseline>` là chênh lệch, so từng cặp trên cùng episode. `p_improve_hv_vs_<baseline>` là xác suất một seed của phương pháp có HV cao hơn một seed của baseline (Agarwal et al., NeurIPS 2021). `--family <tên> <file.npy>` thêm một họ policy không học, ví dụ các heuristic. `<out>_seeds.csv` có số liệu từng seed.

Hành vi và hình cho báo cáo (3 UAV): `eval_trajectories.py` ghi vị trí, công suất, độ phủ từng bước ở 3 góc trọng số trên cùng episode test (cần checkpoint trong `runs/`); `plot_results.py` vẽ đường học, HV có CI, ablation 2×2, front Pareto, quỹ đạo, và ghi `behaviour.csv`, vào `figures/` (không commit).

```bash
python eval_trajectories.py --out results-v1.1/eval_test/trajectories
python plot_results.py --results results-v1.1 --out figures
```

Mỗi nhận định trong `CLAUDE.md` và báo cáo không đến từ `eval_checkpoint.py` có một script kiểm trong `report/scripts/` (danh sách trong `report/README.md`), chạy từ gốc repo.


### Khoá luận: chọn omega lúc chạy (đang phát triển)

`eval_controller.py` chạy policy `ma3_prior` đã train, nhưng omega đổi mỗi bước theo KPI mà nhà mạng đặt ngưỡng (PID-Lagrange, hàng đợi ảo, DPI rút gọn). Script so với các omega cố định và omega cố định tốt nhất chọn sau khi biết kết quả. Không train lại. Chỉnh hệ số trên seed 100000+, báo cáo trên 200000+.

```bash
python eval_controller.py --out evals/ctrl --seed_base 200000 --constraint qos:0.30 --primary energy
# Cả ca 20 episode, nhà mạng nâng ngưỡng lên 0,35 từ episode 10
python eval_controller.py --out evals/ctrl_change --seed_base 200000 --carry --change 10:0.35 --k_i 0.5 --k_p 0
# Phép thử vai trò: một UAV ở góc năng lượng, hai UAV kia ở góc QoS
python eval_trajectories.py --methods ma3_prior ma3_prior_pa --roles --seed_base 100000 --out evals/roles
```

Xem lại một episode trên bản đồ (UAV bay, UE đi, association, UE trượt SLA vì đâu): `visualize/record.py` ghi episode của checkpoint (chỉ đọc env, kết quả không đổi), `--html` ra một trang tự chứa mở thẳng bằng trình duyệt. Thiết kế ở `visualize/architecture.md`.

```bash
python -m visualize.record --methods ma3_prior ma3 --episodes 0 1 --out visualize/out/run1 --html
# Mặc định 3 góc; thêm trọng số bất kỳ (--omega, lặp được) hoặc các trọng số của eval lúc train (--sweep 30)
python -m visualize.record --methods ma3_prior --corners qos --omega 0.6,0.2,0.2 --omega 1,1,1 --out visualize/out/w --html
```

Pilot 300k bước cho kiến trúc mới (chạy trên Kaggle, xem `kaggle/README.md`): `ma3_prior_pa` (mỗi UAV một omega), `ma3_prior_att` (mạng attention), `ma3_prior_att_fleet` (attention, môi trường `ma3_fleet.yaml` có UAV rút giữa chừng).

Các khóa mới cho môi trường v2 và agent, mặc định tắt. Khi tắt, kết quả `v1.1` giữ nguyên từng bit (đã kiểm):

| Khóa | Ở đâu | Ý nghĩa |
|---|---|---|
| `slicing.rb_sharing` | config môi trường | `none`, `idle` (RB của vùng không có UE có tải chia cho vùng khác của cùng UAV), `all` (cả RB mà action để trống) |
| `slicing.per_slice_queues` | config môi trường | Mỗi slice một hàng đợi ở mỗi UAV, thay cho một hàng đợi chung |
| `fleet` (`n_active_min`, `withdraw_prob`, `withdraw_steps`) | config môi trường | Số UAV hoạt động đổi khi reset, một UAV có thể rút giữa episode; UAV rút vẫn giữ chỗ trong quan sát và hành động |
| `objectives` | config môi trường (v2) | Vector reward theo tên, ví dụ `[sla_embb, sla_urllc, sla_mmtc, energy]`: SLA mềm mỗi slice (TIP 1) |
| `action.mode: slice_share` | config môi trường (v2) | Hành động 7 chiều: di chuyển, công suất, tỉ lệ RB mỗi slice; bộ lập lịch trong bước (TIP 2) |
| `shift` | config môi trường (v2) | Ca trực 240 bước kết thúc thật; mỗi ca rút 2–10 vị trí phục vụ; UAV dự phòng bay ra thay ca, UAV về đổi pin, trạm BS60 sạc theo đúng thời gian của DJI (TIP 4, kiểm bằng `py docs/v2/check_tip4.py`) |
| `omega_per_agent` | config train (3 UAV) | Tỉ lệ episode mà mỗi UAV có omega riêng; critic nhận omega của mọi UAV |
| `network` | config train (3 UAV) | `mlp` (mặc định, như paper) hoặc `attention`: một actor chung và critic attention, không phụ thuộc số UAV; checkpoint 3 UAV chạy được với 4 UAV. Chậm hơn khoảng 40% mỗi bước train |
| `omega_switch_prob` | config train (3 UAV) | Xác suất đổi omega ở mỗi bước trong episode |

### 3. Lưu kết quả vào repo

Chép các CSV, và nếu muốn thì các `.npy` của eval, vào `results-v<phiên bản môi trường>/` theo cấu trúc của `results-v1.1/` rồi commit. Không commit checkpoint (`*.pth`) hay replay buffer (`*.npz`, `resume/`); `.gitignore` đã loại các file này.

## Cấu trúc repo

| Đường dẫn | Nội dung |
|---|---|
| `train_moma_ac.py` | Vòng lặp train, eval, ghi log, lưu và resume. |
| `agents_moma_ac.py` | Agent TD3 có điều kiện omega, OLS/GPI-LS/GPI-PD, agent CTDE nhiều UAV. |
| `environment.py` | Mô phỏng mạng: UAV, UE, kênh truyền, cấp phát RB, hàng đợi, reward. |
| `city.py` | Thành phố tường minh theo ITU-R P.1410 và kiểm tra tầm nhìn thẳng (LoS). |
| `eval_checkpoint.py` | Eval cuối với khoảng tin cậy. |
| `aggregate_seeds.py` | Gộp eval cuối của nhiều seed, khoảng tin cậy bootstrap hai tầng. |
| `eval_formation.py` | Heuristic 3 UAV đứng yên thành tam giác. |
| `eval_heuristics.py` | Họ 10 heuristic 3 UAV (bay theo tam giác r = 600 m), cùng episode với `eval_checkpoint.py`. |
| `eval_trajectories.py` | Ghi quỹ đạo và hành vi của các policy 3 UAV ở 3 góc trọng số. |
| `eval_controller.py` | Khoá luận: chọn omega lúc chạy theo ngưỡng KPI, trên policy đã train. |
| `visualize/` | Trình xem episode: bộ ghi `record.py` và trang `viewer.html`. |
| `plot_results.py` | Vẽ các hình của báo cáo từ `results-v1.1/`. |
| `compare_runs.py` | Gộp các seed. |
| `config/train/` | Config train cho từng variant. |
| `config/environment/` | `default.yaml` (1 UAV, khu 1155 m) và `ma3.yaml` (3 UAV, khu 2 km), có nguồn cho từng tham số. |
| `kaggle/` | Notebook, script và hướng dẫn chạy trên Kaggle. |
| `results-v1.1/` | Kết quả của môi trường `v1.1`. |

## Xử lý sự cố

| Triệu chứng | Cách xử lý |
|---|---|
| `Config file not found: config/environment/default.yaml` | Đang không đứng ở thư mục gốc của repo. `cd` về thư mục gốc rồi chạy lại. |
| `ModuleNotFoundError` | Chưa cài `requirements.txt`, hoặc đang dùng nhầm môi trường Python. |
| Lần chạy đầu đứng lâu trước khi train | numba đang biên dịch và ghi cache, chỉ xảy ra một lần. |
| `--resume` vẫn train từ đầu | Sai `--seed` hoặc `--config`, hoặc run chưa từng lưu trạng thái (dừng trước mốc lưu đầu tiên). |
| Hết bộ nhớ | Buffer của `ma3` chiếm khoảng 0,6 GB RAM mỗi run. Giảm số run chạy cùng lúc. |
| Kết quả khác máy bạn một chút | Bình thường: vòng lặp kín rất nhạy với sai số float, nên cùng checkpoint cho HV khác nhau một chút trên CPU, GPU và Kaggle. Dùng `eval_checkpoint.py` (CPU) để có số so sánh được. |
