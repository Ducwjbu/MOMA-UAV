# Train MOMA-AC trên Kaggle

Train 3 variant (`random`, `gpi_ls`, `gpi_pd`), 1M bước, với 1 UAV. Notebook mặc định chạy 1 seed trên GPU; thêm seed bằng cách sửa `SEEDS`. Kaggle giới hạn mỗi session khoảng 12 giờ, nên bộ này tự dừng trước khi hết giờ, lưu trạng thái, và chạy tiếp ở session sau.

## Chuẩn bị

Cần tài khoản Kaggle đã xác minh số điện thoại (Settings → Phone Verification). Không xác minh thì không bật được Internet và GPU.

Tạo file code từ thư mục gốc của repo. Làm lại bước này mỗi khi sửa code:

```powershell
py kaggle/build_bundle.py
```

Lệnh này tạo `kaggle/moma_uav_code.zip`.

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
- `runs/.../pareto_points.csv`: từng điểm đã eval (omega và QoS, energy, fairness), có thể tự lọc Pareto front từ đây.
- `runs/.../checkpoints/`: trọng số mạng.
- `runs/logs/`: log đầy đủ từng run.

Bảng tổng kết ở mốc cuối nằm ở cuối log. Muốn tính lại trên máy: tải các thư mục `saved_models_moma_ac_*` về rồi chạy `py compare_runs.py <các thư mục>`.

## Đọc kết quả thế nào

- So `hv`, `eum`, `eu_front` của các variant, và phải nhìn cả độ lệch giữa các seed. 3 seed chỉ đủ để thấy xu hướng. Paper MOMA-AC dùng 10 seed.
- Hai variant GPI còn có cột `*_actor` là actor thường không dùng GPI. So `hv_actor` với `hv` của `random` để tách tác dụng của cách chọn omega khỏi tác dụng của suy luận GPI.
- `gpi_ls_metrics.csv`: `support_size` (số omega trong M), `max_gap` (cải thiện GPI đo được), và `model_val_nll` với GPI-PD.

## CPU hay GPU

Đo trên laptop này (GTX 1650), 3 tiến trình cùng lúc, tốc độ mỗi tiến trình khi đang update, chưa tính eval:

| Variant | CPU | GPU | 1M bước trên GPU, chưa tính eval |
|---|---|---|---|
| `random` | 28 bước/s | khoảng 39 bước/s | khoảng 7 giờ |
| `gpi_ls` | 21 bước/s | khoảng 35 bước/s | khoảng 8 giờ |
| `gpi_pd` | khoảng 10 bước/s | khoảng 24 bước/s | khoảng 12 giờ |

GPU tiết kiệm thời gian rõ rệt, nhưng tốn quota khoảng 30 giờ mỗi tuần, còn CPU thì không. Khi hết quota hoặc muốn chạy thêm seed bằng CPU: đặt `DEVICE = "cpu"`, `PARALLEL = None` và Accelerator `None`. Nếu đặt `"cuda"` mà không có GPU, run tự chuyển sang CPU và log in `WARNING: no GPU found`.

Số trên đo ở máy này. Máy Kaggle có thể khác, hãy xem tiến độ ở log của session đầu.

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
