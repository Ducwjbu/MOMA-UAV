# Brainstorm hướng khoá luận (3 UAV)

Ghi lại từ buổi thảo luận ngày 2026-11-10, để sau này implement. Khoá luận cần xong khoảng 2027-05. Báo cáo môn học dùng môi trường tag `v1.1`; ghi chú của báo cáo nằm trong `report/`.

**Trạng thái:** đang brainstorm, chưa implement gì. Bài DPI sẽ được gửi lại sau để đọc kỹ.

## Điểm xuất phát

- **Phương pháp hiện tại:** `ma3_prior`, residual policy trên đội hình tam giác cố định, γ 0.9. Trên seed test, HV 0.332 so với 0.176 của `ma3` (paper) và 0.274 của họ heuristic; rất ổn định giữa các seed.
- **Điểm yếu đã biết:**
  - Ở góc năng lượng, `ma3_g90` tốt hơn.
  - Front của `ma3_prior` dồn cục ở vùng ưu tiên QoS.
  - Tam giác cố định quanh tâm, không đổi theo nhu cầu.
  - Mạng chỉ chạy được với đúng 3 UAV, vì thông tin của các UAV khác được ghép cứng vào quan sát.
  - Mới thử trên một kịch bản.
- **Môi trường cần sửa:** hiệu chỉnh SLA eMBB, scheduler chia lại RB thừa, hàng đợi riêng cho từng slice. Xem `report/05_han_che.md`.
- **Ràng buộc:** quota Kaggle có hạn (5 tài khoản, 1M bước ≈ 9.5 giờ, 2 run mỗi phiên), nên tránh train riêng một mô hình cho mỗi kịch bản.

## Ý tưởng 1: chọn trọng số hiệu quả khi train (GPI-LS / OLS)

**Ý tưởng:** thay vì rút ω ngẫu nhiên Dirichlet(1,1,1) mỗi episode, ưu tiên train ở những ω mà policy đang kém nhất so với mức có thể đạt. Nguồn: GPI-PD của Alegre et al., AAMAS 2023.

**Hiện trạng repo:**
- `gpi_ls`, `gpi_pd` và `ols` đã có cho bài 1 UAV (`MOMA_AC_Agent`, `GPILinearSupport`).
- `MOMA_CTDE_Agent` (3 UAV) chỉ hỗ trợ `omega_sampling: random`; các giá trị khác ném `ValueError`.

**Kỳ vọng:** cải thiện các ω ít được học kỹ, như góc năng lượng, và có thể làm front trải rộng hơn. Chỉ số để kiểm chứng: HV, EUM, giá trị ở từng góc.

**Rủi ro:**
1. **GPI-LS dựa vào critic để xếp hạng ω.** Critic của bài 3 UAV từng bị lệch nặng với γ 0.99 (xem memory `ma3-critic-pessimism`), và đỡ hơn với γ 0.9. Nếu critic sai thì GPI-LS ưu tiên sai chỗ.
2. **Chưa có bằng chứng trên môi trường hiện tại:** các so sánh GPI-LS với rút ngẫu nhiên trong repo là trên môi trường cũ.
3. **Tốn bước env không được đếm:** mỗi lần cập nhật support tốn `(|M| + số corner) × gpi_ls_episodes × eval_steps` bước trên `eval_env` (xem Known pitfalls trong CLAUDE.md).
4. **Bài toán là CTDE, actor nhiều đầu ra:** chọn hành động theo GPI (Eq. 21) phải xét hành động chung của cả đội, nên phức tạp và tốn hơn bản 1 UAV.

**Kế hoạch đề xuất:**
- Thử **OLS trước**. OLS xếp hạng corner weight bằng giá trị *đo thật*, không dùng GPI để chọn hành động, nên ít bị critic lệch kéo sai.
- Thí nghiệm: `ma3_prior` + OLS so với `ma3_prior` (rút ngẫu nhiên), 5 seed, eval trên seed test riêng. Tốn khoảng một đợt Kaggle.
- Nếu OLS giúp được thì mới thử GPI-LS hoặc GPI-PD.

**Việc cần implement:**
- Cho `MOMA_CTDE_Agent` dùng `GPILinearSupport` để rút ω theo episode.
- Cập nhật support định kỳ bằng `eval_omega` của agent CTDE.
- Update hai nửa batch (ω hiện tại và `w' ~ M`) như bản 1 UAV.
- Config mới, ví dụ `moma_ac_ma3_prior_ols.yaml`.
- Smoke test theo hướng dẫn trong CLAUDE.md.

## Ý tưởng 2: DPI, agent tự suy ra cần ưu tiên gì

**Ý tưởng của bạn:** các agent tự suy ra lúc này hệ thống cần chuyển sang preference ω nào, thay vì người vận hành phải chỉnh tay.

**Vấn đề cốt lõi:** phải định nghĩa "cần" theo tiêu chí gì. Nếu agent chọn ω để tối đa điểm của chính nó, nó sẽ chọn mục tiêu dễ nhất. Ví dụ: chọn năng lượng rồi đứng yên, tắt máy phát. Paper DPI cũng thừa nhận nghiệm suy biến này (Cao et al. 2026, Sec. 3.3).

### Tài liệu liên quan

- **Cao, Quan, Zhang, Wang.** "Learning What Matters Now: Dynamic Preference Inference under Contextual Shifts". arXiv:2603.22813, 2026-03 (preprint), code tại https://github.com/XianweiC/DPI.
  - Coi ω là biến ẩn: ω = softmax(z), với z ~ q(z | lịch sử gần), encoder hồi quy.
  - Huấn luyện bằng ELBO, trong đó "evidence" là `<ω, G(e)>` do chính critic của agent ước lượng (Eq. 1–3).
  - Chống suy biến bằng KL về N(0, I), loss hướng `1 − cos(ω, G)` (Eq. 8) và loss ổn định (Eq. 9). Lúc ra quyết định: rút K ứng viên ω, chọn ω tốt nhất theo critic (envelope, Eq. 4). Tổng loss ở Eq. 10, trên nền PPO.
  - Một agent, môi trường Queue và Maze (rời rạc), MO-HalfCheetah (liên tục). Các môi trường đều có tiêu chí hoàn thành nhiệm vụ: không đổi preference thì không xong việc.
  - Kết quả không áp đảo baseline ở mọi chỉ số. Phần kết quả mới đọc qua bản trích; cần đọc lại bản đầy đủ khi bạn gửi bài.
- **Abels et al.** "Dynamic Weights in Multi-Objective Deep RL". ICML 2019, PMLR 97:11–20. Một mạng có điều kiện theo ω chạy được khi ω đổi theo thời gian, kèm Diverse Experience Replay. Đây là cơ sở cho việc dùng lại checkpoint hiện có với ω thay đổi.
- **Tessler, Mankowitz, Mannor.** "Reward Constrained Policy Optimization". ICLR 2019. Hệ số phạt (nhân tử Lagrange) tự điều chỉnh theo mức vi phạm ràng buộc. Đây là cơ sở cho cách A bên dưới.
- **Lu, Mannion, Mason.** "Inferring Preferences from Demonstrations in MORL". Neural Computing & Applications 2024, doi:10.1007/s00521-024-10412-x (arXiv:2409.20258). Suy ra ω từ thao tác hoặc demonstration của người.
- **DPIN.** "Dynamic preference inference network: Improving sample efficiency for MORL by preference estimation". Knowledge-Based Systems. Ước lượng preference mà một trajectory tối ưu theo, để tăng hiệu quả mẫu; mục đích khác ý tưởng của bạn. Trang ScienceDirect bị chặn (403), chưa đọc được.

### Ba cách định nghĩa "cần"

| Cách | "Cần" nghĩa là | Cần train lại? | Đánh giá |
|---|---|---|---|
| **A. Theo mục tiêu vận hành** (khuyên dùng) | Nhà mạng đặt mục tiêu một lần, ví dụ URLLC ≥ x% SLA hay pin đủ cho hết ca. Khi một mục tiêu sắp bị vi phạm, ω tự nghiêng về mục tiêu đó (cập nhật kiểu nhân tử Lagrange, RCPO). | Không: thêm một tầng điều khiển ω trên checkpoint `ma3_prior` | Dễ giải thích; hợp nguyên tắc "người quyết định mục tiêu, agent quyết định cách cân bằng" |
| B. Học từ người vận hành | Bắt chước cách người vận hành chỉnh ω theo tình huống | Có | Thiếu dữ liệu thao tác của người vận hành |
| C. Học biến ẩn như paper DPI | ω suy ra từ critic và lịch sử | Có | Cần thêm tiêu chí hoàn thành nhiệm vụ; mới thử với một agent; critic của ta từng lệch |

### Câu hỏi mở (bạn trả lời sau)

- "Hệ thống cần" dựa trên tín hiệu nào? Ứng viên:
  - tỉ lệ đạt SLA theo slice (đã có trong `info['sla_*']`);
  - mức pin hoặc năng lượng còn lại so với năng lượng cần để quay về;
  - số người dùng hoặc tải tăng đột biến;
  - độ phủ.
- Mục tiêu vận hành cụ thể là gì? Ngưỡng SLA bao nhiêu, ca bay bao lâu?
- ω đổi ở cấp nào: cả đội dùng chung một ω (reward hiện tại là reward chung của đội), hay mỗi UAV một ω?
- Người vận hành có quyền ghi đè ω không?

### Thiết kế thí nghiệm cho cách A (không cần Kaggle)

- **So sánh:** ω cố định (vài giá trị), ω theo lịch heuristic, và DPI theo mục tiêu vận hành. Tất cả dùng chung checkpoint `ma3_prior`.
- **Kịch bản:** kịch bản hiện tại, và kịch bản có thay đổi giữa episode (tan concert: đám đông dồn về một góc; tải URLLC tăng; pin hao nhanh do gió mạnh).
- **Chỉ số:** tỉ lệ thời gian đạt từng mục tiêu vận hành, 3 reward, số lần ω dao động (độ ổn định).
- **Lưu ý:** episode hiện dài 18 bước (18 phút), nên thay đổi phải xảy ra trong khung thời gian đó.

## Ý tưởng 3: chạy được ở kịch bản và số UAV mới mà không train nhiều kịch bản

Tổng quan về chủ đề này: Kirk, Zhang, Grefenstette, Rocktäschel, "A Survey of Zero-shot Generalisation in Deep RL", JAIR 76:201–264, 2023.

**Lưu ý về quota:** môi trường đã tạo bản đồ thành phố và bản đồ người dùng mới ở mỗi `reset`, nên đa dạng bản đồ trong một lần train gần như miễn phí. Thứ tốn quota là train riêng mô hình cho mỗi kịch bản.

| Cách | Ý tưởng | Nguồn |
|---|---|---|
| **3a. Prior theo nhu cầu** (khuyên dùng) | Đỉnh đội hình = weighted k-means trên mật độ người dùng quan sát được, tính lại khi chạy. Khi số UAV đổi thì gán lại đỉnh bằng thuật toán Hungarian. Actor chỉ nhìn tương đối (độ lệch so với đỉnh, người dùng xung quanh). | k-means hay được dùng làm bước đặt vị trí UAV trong tài liệu; chưa thấy paper kiểm tra đúng tổ hợp này trên kịch bản khác lúc train, nên có thể là điểm mới |
| 3b. Mạng không phụ thuộc số UAV | Mean embedding, mean field, transformer, hoặc GNN cho thông tin của các UAV khác | Hüttenrauch, Šošić, Neumann, JMLR 20(54), 2019; Yang et al., ICML 2018; Hu et al. (UPDeT), ICLR 2021; tổng quan GNN cho UAV, Drones 10(7):548, 2026 |
| 3c. Thích nghi khi triển khai | Tự giám sát không cần reward (PAD), hoặc fine-tune ngắn | Hansen et al., ICLR 2021 |
| 3d. Nếu vẫn cần đa dạng thì làm rẻ | Ưu tiên luyện lại các kịch bản "đáng học" | Jiang, Grefenstette, Rocktäschel (PLR), ICML 2021, PMLR 139:4940–4950 |

**Giả định cần hỏi thầy:** 3a cần biết người dùng tập trung ở đâu. Nhà mạng ước lượng được điều này từ đo đạc mạng, nhưng nên xác nhận giả định này được chấp nhận.

**Bước đầu không tốn quota:** chạy checkpoint `ma3_prior` hiện có trên kịch bản đám đông dồn góc (và 2 hoặc 4 UAV, nếu sửa được phần quan sát) để đo mức sụt giảm. Đây là số liệu "trước cải tiến".

## Thứ tự đề xuất

0. Tạo nhánh `thesis` từ `duc`; giữ tag `v1.1` cho báo cáo môn học.
1. Sửa môi trường lên v2 (eMBB, scheduler, hàng đợi theo slice), vì mọi lần sửa môi trường đều phải train lại.
2. Thêm kịch bản thay đổi giữa episode (tan concert, tải tăng, gió mạnh), dùng chung cho ý tưởng 2 và 3.
3. **DPI cách A** trên checkpoint hiện có: không tốn quota.
4. **OLS cho CTDE:** một đợt Kaggle.
5. **Prior theo nhu cầu, rồi mạng không phụ thuộc số UAV:** một hoặc hai đợt Kaggle.
6. Quét γ có hệ thống với phương pháp cuối cùng: γ ∈ {0.5, 0.8, 0.9, 0.95, 0.99}, seed validation riêng, cùng ngân sách cho mọi phương pháp (Andrychowicz et al., ICLR 2021, Sec. 3.6; Eimer, Lindauer, Raileanu, ICML 2023, Sec. 6).

**Ba ý tưởng bổ trợ nhau:**
- DPI giúp agent phản ứng khi tình huống đổi.
- OLS hoặc GPI-LS giúp policy tốt đều ở mọi ω, để khi DPI chuyển ω thì policy ở ω mới vẫn tốt.
- Prior theo nhu cầu và mạng không phụ thuộc số UAV giúp chạy được ở bản đồ và số UAV mới.

## Link

- DPI: https://arxiv.org/html/2603.22813
- GPI-PD: https://arxiv.org/abs/2301.07784
- Dynamic weights: https://arxiv.org/abs/1809.07803
- RCPO: https://arxiv.org/abs/1805.11074
- DWPI: https://arxiv.org/abs/2409.20258
- Survey ZSG: https://jair.org/index.php/jair/article/view/14174
- Mean embedding: https://arxiv.org/abs/1807.06613
- Mean field: https://arxiv.org/abs/1802.05438
- UPDeT: https://arxiv.org/abs/2101.08001
- PAD: https://arxiv.org/abs/2007.04309
- PLR: https://arxiv.org/abs/2010.03934
- GNN UAV review: https://doi.org/10.3390/drones10070548
