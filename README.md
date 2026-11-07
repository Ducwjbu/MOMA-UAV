# MOMA-UAV

Học tăng cường đa mục tiêu điều khiển UAV làm trạm gốc, cắt lát mạng (network slicing) cho ba slice eMBB, URLLC và mMTC. Agent là TD3 có điều kiện theo vector trọng số omega. Agent học một Pareto front trên ba mục tiêu: QoS, năng lượng và fairness.

Repo có hai bài toán:

- **1 UAV**: so sánh các cách chọn omega khi train. `random` là baseline (MOMA-AC gốc, lấy mẫu Dirichlet đều), so với `ols`, `gpi_ls` và `gpi_pd`.
- **3 UAV (`ma3`)**: MOMA-TD3 của paper MOMA-AC, huấn luyện tập trung, thực thi phân tán (CTDE). Hiện chỉ có lấy mẫu omega đều. Bản cải tiến `ma3_prior` cho actor học phần lệch (residual) trên một đội hình nền cố định, với γ = 0,9; `ma3` giữ làm baseline.

Mô tả kỹ thuật đầy đủ (môi trường, reward, agent, các lỗi đã biết) nằm trong [CLAUDE.md](CLAUDE.md). Hướng dẫn chạy trên Kaggle nằm trong [kaggle/README.md](kaggle/README.md).

## Phiên bản môi trường

Môi trường `v1.0` (git tag `v1.0`) có thành phố tường minh theo ITU-R P.1410, suy hao xuyên tường, shadowing tương quan, fast fading, gió Hà Nội, traffic theo chuẩn 3GPP, các trạm bên đường (RSU) cho URLLC, và QoS theo SLA. Nguồn của từng tham số được ghi trong comment của `config/environment/*.yaml`.

Sau `v1.0` môi trường đổi hai lần, nên mọi kết quả trong `results-v1/` đều phải chạy lại:

- **Pin dự trữ để bay về:** theo cơ chế Low Battery RTH của DJI M300. Mỗi episode dài 18 bước với 3 UAV, 19 bước với 1 UAV.
- **Mô hình lưu lượng mới:** bản đồ mật độ lưu lượng log-normal của Lee et al. 2014 (`traffic_map.py`, mục `traffic_map` trong config) thay cho các hotspot. Mỗi episode sinh một bản đồ mới. UE eMBB xuất hiện theo bản đồ, UE mMTC xuất hiện đều.

Hai thay đổi này sẽ được gắn tag `v1.1`.

**Mọi thay đổi môi trường đều làm kết quả cũ mất giá trị so sánh**, vì thang reward thay đổi. Vì vậy:

- Chỉ so những run cùng phiên bản môi trường.
- `results-v0/` là kết quả của môi trường cũ, chỉ dùng để kiểm tra rằng train chạy được. Không đưa các số này vào bài báo.
- `results-v1/` là kết quả của môi trường `v1.0`.

Muốn sửa môi trường thì bàn trước với cả nhóm. Nếu sửa, tạo tag mới (`v1.1`, ...) và chạy lại toàn bộ.

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
- Config này dùng γ = 0,9. Với γ = 0,99, giá trị của critic trôi xa giá trị thật và kéo actor rời đội hình.
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

**Chia seed trong nhóm.** Mỗi người nhận seed riêng để không chạy trùng, và ghi lại ai chạy seed nào. Seed 0 của `ma3` đã chạy xong (`results-v1/`).

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

Eval chạy trên các episode chưa từng dùng khi train (seed 100000 trở đi), trên CPU để lặp lại được, có khoảng tin cậy 95% bootstrap:

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

Lệnh này in mean ± std qua các seed ở mốc eval cuối, và ghi `comparison.csv` cho mọi mốc eval. Mỗi thư mục là `save_dir` của một variant, chứa các `seed_*/`.

### 3. Lưu kết quả vào repo

Chép các CSV, và nếu muốn thì các `.npy` của eval, vào `results-v1/` theo cấu trúc sẵn có rồi commit. Không commit checkpoint (`*.pth`) hay replay buffer (`*.npz`, `resume/`); `.gitignore` đã loại các file này.

## Cấu trúc repo

| Đường dẫn | Nội dung |
|---|---|
| `train_moma_ac.py` | Vòng lặp train, eval, ghi log, lưu và resume. |
| `agents_moma_ac.py` | Agent TD3 có điều kiện omega, OLS/GPI-LS/GPI-PD, agent CTDE nhiều UAV. |
| `environment.py` | Mô phỏng mạng: UAV, UE, kênh truyền, cấp phát RB, hàng đợi, reward. |
| `city.py` | Thành phố tường minh theo ITU-R P.1410 và kiểm tra tầm nhìn thẳng (LoS). |
| `eval_checkpoint.py` | Eval cuối với khoảng tin cậy. |
| `eval_formation.py` | Heuristic 3 UAV đứng yên thành tam giác. |
| `compare_runs.py` | Gộp các seed. |
| `config/train/` | Config train cho từng variant. |
| `config/environment/` | `default.yaml` (1 UAV, khu 1155 m) và `ma3.yaml` (3 UAV, khu 2 km), có nguồn cho từng tham số. |
| `kaggle/` | Notebook, script và hướng dẫn chạy trên Kaggle. |
| `results-v0/`, `results-v1/` | Kết quả theo phiên bản môi trường. |

## Xử lý sự cố

| Triệu chứng | Cách xử lý |
|---|---|
| `Config file not found: config/environment/default.yaml` | Đang không đứng ở thư mục gốc của repo. `cd` về thư mục gốc rồi chạy lại. |
| `ModuleNotFoundError` | Chưa cài `requirements.txt`, hoặc đang dùng nhầm môi trường Python. |
| Lần chạy đầu đứng lâu trước khi train | numba đang biên dịch và ghi cache, chỉ xảy ra một lần. |
| `--resume` vẫn train từ đầu | Sai `--seed` hoặc `--config`, hoặc run chưa từng lưu trạng thái (dừng trước mốc lưu đầu tiên). |
| Hết bộ nhớ | Buffer của `ma3` chiếm khoảng 0,6 GB RAM mỗi run. Giảm số run chạy cùng lúc. |
| Kết quả khác máy bạn một chút | Bình thường: vòng lặp kín rất nhạy với sai số float, nên cùng checkpoint cho HV khác nhau một chút trên CPU, GPU và Kaggle. Dùng `eval_checkpoint.py` (CPU) để có số so sánh được. |
