# Train MOMA-AC trên Kaggle

Train 4 variant (`random`, `ols`, `gpi_ls`, `gpi_pd`), 1M bước, với 1 UAV, và `ma3` (3 UAV, xem phần cuối). Notebook mặc định chạy 1 seed trên GPU; thêm seed bằng cách sửa `SEEDS`. Kaggle giới hạn mỗi session khoảng 12 giờ, nên bộ này tự dừng trước khi hết giờ, lưu trạng thái, và chạy tiếp ở session sau.

## Chuẩn bị

Cần tài khoản Kaggle đã xác minh số điện thoại (Settings → Phone Verification). Không xác minh thì không bật được Internet và GPU.

Tạo file code từ thư mục gốc của repo. Làm lại bước này mỗi khi sửa code:

```powershell
py kaggle/build_bundle.py
```

Lệnh này tạo `kaggle/moma_uav_code.zip`.

> **Đổi môi trường (tag `v1.0` trở đi).** Kết quả và trạng thái resume của môi trường cũ không dùng lại được: quan sát đổi chiều (83 với 1 UAV, 41 với `ma3`) và reward đổi thang. Khi chạy với code mới, upload Dataset mới (hoặc **New Version**) và dùng notebook mới, hoặc gỡ hết các Input là output notebook cũ. Lần chạy đầu, đầu log phải có `No earlier runs attached`. Nếu thấy `[Restore] ...` tức là đang gắn output cũ: dừng lại và gỡ Input đó.

## Bước 1: Upload code thành Dataset

1. Vào [kaggle.com/datasets](https://www.kaggle.com/datasets) → **New Dataset**.
2. Kéo file `moma_uav_code.zip` vào, đặt tên ví dụ `moma-uav-code`.
3. Để chế độ **Private** → **Create**.

Kaggle thường tự giải nén zip. Notebook xử lý được cả hai trường hợp, đã giải nén hay chưa.

## Bước 2: Tạo notebook

1. Vào [kaggle.com/code](https://www.kaggle.com/code) → **New Notebook**.
2. **File → Import Notebook** → chọn `kaggle/moma_ac_kaggle.ipynb`.
3. Thanh bên phải, mục **Input** → **Add Input** → **Your Work → Datasets** → chọn `moma-uav-code`.
4. Thanh bên phải, mục **Session options**:
   - **Accelerator**: GPU. Nếu có T4 x2 thì nên chọn, các run sẽ tự chia đều cho 2 GPU. Xem thêm phần "CPU hay GPU".
   - **Internet**: bật. Chỉ cần nếu máy Kaggle chưa có sẵn `numba`.
5. Đổi tên notebook ở góc trên, ví dụ `moma-ac-train`.

## Bước 3: Chạy thử

Để nguyên `SMOKE_TEST = True` ở ô cài đặt, bấm **Run All**. Sau vài phút, phải thấy dòng:

```
[Kaggle] SMOKE TEST PASSED
```

Ô thứ hai phải hiện tên GPU, và dòng `[Kaggle] ... runs to do` cho biết số GPU tìm thấy. Nhớ dừng session sau khi thử xong, vì session đang mở vẫn tính vào quota GPU. Các chỉ số trong lần thử không có ý nghĩa, vì chỉ train 1500 bước.

Nếu có lỗi, xem phần "Xử lý sự cố".

## Bước 4: Train thật

1. Đổi ô cài đặt thành `SMOKE_TEST = False`.
2. Góc trên bên phải: **Save Version** → chọn **Save & Run All (Commit)** → **Save**.

Bạn có thể tắt trình duyệt, tắt máy. Notebook vẫn chạy trên Kaggle. Theo dõi ở trang notebook, tab **Versions** → mở version đang chạy → **Logs**. Cứ 15 phút có một dòng tiến độ.

> Đừng dùng **Run All** cho train thật. Chế độ tương tác cần tab trình duyệt mở, và output chỉ được lưu lại khi commit.

## Bước 5: Chạy tiếp ở session sau

Với 1M bước, một run có thể cần hơn một session, nên đây là bước bình thường chứ không phải sự cố. Khi version chạy xong (khoảng 11 giờ), mở tab **Output** hoặc **Logs**, xem cuối log:

- `ALL RUNS DONE`: xong, sang Bước 6.
- `Not finished: ...`: làm tiếp như sau.

1. Mở lại notebook, **Edit**.
2. **Add Input** → **Your Work → Notebooks** → chọn chính notebook `moma-ac-train`. Từ lần thứ ba: nếu input đang gắn là version cũ, gỡ ra rồi gắn lại để lấy version mới nhất.
3. **Save Version → Save & Run All (Commit)** lần nữa.

Đầu log phải có các dòng `[Restore] ...: in progress, N evals`. **Nếu thấy `No earlier runs attached` tức là chưa gắn đúng output, và các run sẽ train lại từ đầu.** Khi đó dừng version này và kiểm tra lại input.

Khi resume, run bắt đầu một episode mới vì trạng thái env không được lưu. Buffer, mạng, optimizer, model, RNG và log đều được giữ nguyên.

## Bước 6: Lấy kết quả

Trang notebook, tab **Output** của version cuối:

- `comparison.csv`: HV, EUM, EU và Cardinality của từng variant ở mọi mốc eval, mean và std qua các seed.
- `runs/saved_models_moma_ac_<variant>/seed_<n>/eval_metrics.csv`: số liệu từng run.
- `runs/.../pareto_points.csv`: từng điểm đã eval (omega, QoS, energy, fairness, và `sla_embb`, `sla_urllc`, `sla_mmtc`: tỷ lệ UE đạt đủ SLA của slice), có thể tự lọc Pareto front từ đây.
- `runs/.../checkpoints/`: trọng số mạng.
- `runs/logs/`: log đầy đủ từng run.

Bảng tổng kết ở mốc cuối nằm ở cuối log. Muốn tính lại trên máy: tải các thư mục `saved_models_moma_ac_*` về rồi chạy `py compare_runs.py <các thư mục>`.

## Đọc kết quả thế nào

- So `hv`, `eum`, `eu_front` của các variant, và phải nhìn cả độ lệch giữa các seed. 3 seed chỉ đủ để thấy xu hướng. Paper MOMA-AC dùng 10 seed.
- Hai variant GPI còn có cột `*_actor` là actor thường không dùng GPI. So `hv_actor` với `hv` của `random` để tách tác dụng của cách chọn omega khỏi tác dụng của suy luận GPI.
- `ols` chọn omega theo support M như GPI-LS nhưng luôn hành động bằng actor thường, nên chỉ có cột `hv`, không có `*_actor`. So `hv` của `ols` với `random` để xem tác dụng của việc chọn omega, và với `hv` của `gpi_ls` để xem có GPI thì tốt hơn hay kém đi.
- `eval_metrics.csv` có tỷ lệ UE đạt đủ SLA theo slice: `avg_sla_*` (trung bình qua các omega) và `wqos_sla_*` (ở omega = (1, 0, 0), chỉ ưu tiên QoS). Slice không có tải ở mọi bước thì ra `nan`.
- Số liệu cuối cho bài báo: tải checkpoint 1M về và chạy `eval_checkpoint.py` (20 episode riêng, CI bootstrap, xem `CLAUDE.md`). Eval lúc train chỉ 3 episode, chỉ để xem đường học.
- `gpi_ls_metrics.csv`: `support_size` (số omega trong M), `max_gap` (với GPI là cải thiện đo được, với `ols` là khoảng cách tới cận trên lạc quan, nên không so được giữa hai loại), và `model_val_nll` với GPI-PD.

## CPU hay GPU

Đo trên laptop này (GTX 1650) với **môi trường cũ** (trước `v1.0`), 3 tiến trình cùng lúc, tốc độ mỗi tiến trình khi đang update, chưa tính eval. Môi trường `v1.0` có thành phố, fading và gió nên chậm hơn: riêng phần env khoảng 7 ms/bước với 1 UAV (khoảng 2 giờ cho 1M bước, CPU laptop), nên tổng thời gian sẽ dài hơn bảng này. Hãy xem tiến độ ở log của session đầu.

| Variant | CPU | GPU | 1M bước trên GPU, chưa tính eval |
|---|---|---|---|
| `random` | 28 bước/s | khoảng 39 bước/s | khoảng 7 giờ |
| `gpi_ls` | 21 bước/s | khoảng 35 bước/s | khoảng 8 giờ |
| `gpi_pd` | khoảng 10 bước/s | khoảng 24 bước/s | khoảng 12 giờ |

GPU tiết kiệm thời gian rõ rệt, nhưng tốn quota khoảng 30 giờ mỗi tuần, còn CPU thì không. Khi hết quota hoặc muốn chạy thêm seed bằng CPU: đặt `DEVICE = "cpu"`, `PARALLEL = None` và Accelerator `None`. Nếu đặt `"cuda"` mà không có GPU, run tự chuyển sang CPU và log in `WARNING: no GPU found`.

Số trên đo ở máy này. Máy Kaggle có thể khác, hãy xem tiến độ ở log của session đầu. `ols` chưa được đo riêng. Nó update bằng cùng cách với `gpi_ls` nhưng không tính GPI khi chọn hành động và chỉ eval một lần mỗi mốc, nên không chậm hơn `gpi_ls`.

## Thêm variant cho seed đã chạy

Cập nhật code trước: chạy `py kaggle/build_bundle.py`, rồi vào Dataset `moma-uav-code` → **New Version** → upload `moma_uav_code.zip` mới. Sau đó trong notebook:

1. **Add Input** → output của version đã train, để run cũ không bị train lại.
2. Giữ đủ các variant trong `VARIANTS`, để `comparison.csv` có đủ bảng so sánh. Run đã có `DONE` sẽ được bỏ qua.
3. Đầu log phải có `[Restore] ...: DONE` cho từng run cũ.

## Chạy 3 UAV (`ma3`)

`ma3` là MOMA-TD3 của paper trên 3 UAV, chỉ lấy mẫu omega đều. `ma3_prior` là bản cải tiến: actor học phần lệch trên một đội hình nền cố định, γ = 0,9 (xem README chính). Hai variant dùng chung môi trường `config/environment/ma3.yaml`: khu 2 km, 31 UE eMBB, 60 UE mMTC và 26 trạm bên đường (RSU, slice URLLC), episode 18 bước (pin phải còn đủ để bay về trạm mặt đất). Nó không nằm trong variant mặc định.

1. Nên dùng một notebook riêng (cách tạo như Bước 2). Đặt `VARIANTS = ["ma3_prior"]` (bản cải tiến, ưu tiên chạy trước), `["ma3"]` (baseline), hoặc cả hai, và `SEEDS` theo nhu cầu. Hai ablation là `ma3_g90` (MOMA-TD3 với γ 0,9) và `ma3_prior_g99` (đội hình nền với γ 0,99). Mọi config 3 UAV đã tự trỏ tới `config/environment/ma3.yaml`. Có thể để chung với các variant 1 UAV, nhưng run `ma3` chậm hơn nhiều nên sẽ chiếm chỗ lâu.
2. Kết quả nằm ở `runs/saved_models_moma_ac_ma3/seed_<n>/` (hoặc `saved_models_moma_ac_ma3_prior`). `eval_metrics.csv` và `step_metrics.csv` có thêm `coverage`, `handovers`, `backhaul_limited` (tỷ lệ UAV vượt dung lượng backhaul), `min_uav_distance` (khoảng cách nhỏ nhất thực sự giữa hai UAV) và `mean_uav_distance` (trung bình theo step của cặp gần nhất). Các cột SLA theo slice giống run 1 UAV.
3. So với baseline không cần train: `py eval_checkpoint.py --formation --out eval_triangle` cho 3 UAV đứng yên thành tam giác trên cùng các episode eval với `eval_checkpoint.py` của checkpoint.

Tốc độ: riêng phần env của `ma3` khoảng 24 ms/bước trên CPU laptop, tức khoảng 7 giờ cho 1M bước chưa tính cập nhật mạng, nên một run gần như chắc chắn cần 2 session (Bước 5). Chưa đo trên Kaggle, hãy xem tiến độ ở log của session đầu. Mỗi run chưa xong chiếm nhiều đĩa hơn run 1 UAV, vì buffer lưu quan sát và hành động chung của cả đội.

## Chia việc nếu chạy không kịp

Mỗi session chạy cùng lúc tối đa `PARALLEL` run, các run còn lại chờ đến khi có chỗ trống hoặc tới session sau. Nếu muốn nhanh hơn:

- Giảm `SEEDS` xuống `[1]` trong lượt đầu để có kết quả sớm, rồi thêm seed sau. Run đã xong không bị chạy lại.
- Chia theo notebook: mỗi notebook một phần `SEEDS`, chạy cùng lúc nếu Kaggle cho phép. Lúc gom kết quả, gắn tất cả output làm input của một notebook rồi chạy với đủ `SEEDS`.

## Xử lý sự cố

| Triệu chứng | Cách xử lý |
|---|---|
| `Không thấy code` | Chưa gắn Dataset, làm lại Bước 2 mục 3 |
| Lỗi khi `pip install numba` | Chưa bật Internet hoặc chưa xác minh số điện thoại |
| `FAILED <variant> seed <n>` | Log in ngay dưới dòng đó. Run khác vẫn chạy tiếp, run lỗi sẽ thử lại ở session sau |
| `NO RESUME STATE` | Run chưa kịp lưu lần nào trong session trước, nên train lại từ đầu |
| Version bị dừng vì quá giờ | Có thể mất output. Giảm `MAX_HOURS`, ví dụ còn 9 |
| Đầy đĩa | Mỗi run chưa xong chiếm tới khoảng 400 MB. Giảm số run mỗi notebook |

## Giới hạn của Kaggle

Các con số về quota lấy từ nguồn không chính thức và có thể đã đổi. Xem giá trị hiện tại trong trang notebook và trong [tài liệu của Kaggle](https://www.kaggle.com/docs/notebooks). Nếu session ngắn hơn 12 giờ, giảm `MAX_HOURS` cho phù hợp. `MAX_HOURS` mặc định 10.5 đã chừa chỗ cho eval đang chạy dở, lưu file và upload output.
