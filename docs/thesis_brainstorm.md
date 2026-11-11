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

## Cập nhật 2026-11-11: quyết định và tài liệu mới

Sơ đồ kiến trúc bản 1: https://claude.ai/artifact/QoM1msQLpvBDc9oFWBK8Hd (riêng tư; chia sẻ bằng nút Share).

### Quyết định

- **Hướng "cách 2":** cả đội có một mục tiêu chung do nhà mạng đặt. ω_i của từng UAV là **vai trò** để phục vụ mục tiêu chung đó (ví dụ một UAV lo QoS ở điểm nóng, một UAV lo pin). Đây **không phải** "mỗi UAV một lợi ích riêng" theo nghĩa trò chơi.
- Mỗi UAV một ω_i; người vận hành được ghi đè.
- Tín hiệu kích hoạt và ngưỡng: bàn với thầy.
- Bài tham khảo chính cho ω riêng mỗi agent: **PCMA** (bài bạn đã đọc trước đó).

### DPI: đã đọc bản đầy đủ (Cao et al., ICLR 2026, arXiv:2603.22813)

- **Nơi đăng:** đã được nhận ở ICLR 2026, không chỉ là preprint như ghi ở trên.
- **DPI suy ra ω từ return mà critic của chính agent dự đoán** (Eq. 1–4, 8). Bài viết: "discourages caring about objectives that are unattainable at the moment" (Sec. 3.3). Vì vậy ω dịch về mục tiêu đang dễ đạt.
  - Ví dụ Fig. 1 của bài: người xếp hàng bỏ "đạo đức" để chen hàng khi đói.
  - Với UAV, điều này đi ngược nhu cầu: khi tan concert, QoS khó đạt nhất nhưng cũng cần nhất.
- **Baseline luật viết tay thắng DPI** ở cả ba môi trường. "Dense Oracle" có quyền xem tín hiệu sự kiện (Table 4, 5):

  | Môi trường | Dense Oracle (MER / SR) | DPI (MER / SR) |
  |---|---|---|
  | Maze | 40.3 / 62.9% | 30.2 / 59.0% |
  | Liên tục | 67.0 / 99.98% | 42.1 / 81.0% |

  Nhà mạng nhìn thấy SLA và pin, nên điều khiển theo luật (cách A) là lựa chọn hợp lý.
- **Trở ngại riêng của bài UAV:**
  - Reward năng lượng âm: loss căn hướng Eq. 8 đẩy trọng số năng lượng về 0.
  - Critic của ta từng bị lệch.
  - DPI dùng PPO một agent, ta dùng TD3 nhiều agent.
- **Kết luận:** DPI gốc chỉ dùng làm phương án so sánh.

### Hai bài đã đọc khác

- **Toure, Tsilimantos, Esrafilian, Kountouris**, "Optimizing Energy and Data Collection in UAV-aided IoT Networks using Attention-based MORL", arXiv:2601.14092, 2026.
  - Trạng thái là tập token; self-attention không positional encoding; MaxPool trên token thiết bị; ω là một token riêng.
  - Train với K = 6 thiết bị, test với 3–10 không cần train lại (Fig. 6). Ở w = (0, 1), mạng chủ yếu chú ý vào token ω (Fig. 7).
  - Bài làm với một UAV, hành động rời rạc, SAC; mô hình năng lượng cho hover rẻ hơn bay, ngược với Zeng.
  - **Dùng cho:** actor và critic không phụ thuộc số UAV.
- **Lu, Herman, Yu**, "Multi-Objective RL: Convexity, Stationarity and Pareto Optimality" (CAPQL), ICLR 2023.
  - Với policy ngẫu nhiên dừng, tập giá trị là lồi (Prop. 3), nên tổng có trọng số đạt mọi điểm Pareto (Prop. 4).
  - Policy tất định chỉ đạt các đỉnh, và ánh xạ ω → nghiệm không liên tục (Fig. 6, Remark 4). Cộng entropy thì ánh xạ liên tục (Prop. 5).
  - **Dùng cho:** ta đang dùng TD3 tất định. Entropy còn làm hàm đối ngẫu trơn (xem mục toán bên dưới), nên nó thuộc thiết kế chính chứ không chỉ là ablation.
  - **Lưu ý:** định lý giả định một agent, tầm nhìn vô hạn, MDP ergodic. Policy dạng tích của nhiều agent có thể không giữ tính lồi.

### Cơ sở toán học (đề xuất, chưa kiểm hết)

- **(a) Lagrange = tổng có trọng số.**
  - Bài toán: tối ưu mục tiêu chính với điều kiện V_k ≥ c_k. Hàm Lagrange là `L = ⟨e₀ + λ, V⟩ − λ·c`, tức tổng có trọng số với `ω = (e₀ + λ)/‖e₀ + λ‖₁`.
  - Vì vậy policy có điều kiện theo ω chính là lời giải của hàm Lagrange với mọi λ. Dịch ω theo λ là tăng dần theo gradient ở miền đối ngẫu, **không cần train lại**.
  - Dựa vào: Paternain, Chamon, Calvo-Fullana, Ribeiro, "Constrained RL Has Zero Duality Gap", NeurIPS 2019; CAPQL Prop. 4; Ying, Ding, Lavaei, "A Dual Approach to CMDPs with Entropy Regularization", AISTATS 2022 (có entropy thì hàm đối ngẫu trơn, hội tụ Õ(1/T)).
- **(b) Hàng đợi ảo.**
  - Cập nhật `λ_k ← max(0, λ_k + η(ngưỡng_k − đo_k))` là hàng đợi ảo của drift-plus-penalty (M. J. Neely, *Stochastic Network Optimization with Application to Communication and Queueing Systems*, Morgan & Claypool, 2010).
  - Với C-additive approximation: ràng buộc thoả tính trung bình theo thời gian; mục tiêu cách tối ưu O(1/V), hàng đợi O(V). Có bản cho động học Markov (Huang & Neely, arXiv:1008.0200).
  - Phải tự chuyển từ quyết định từng bước sang policy nhiều bước: coi mỗi lần giữ ω là một khung và dùng kết quả cho hệ renewal của Neely. Cần đối chiếu sách gốc.
- **(c) Exponential tilting** (tự suy ra, chỉ đúng cho posterior chính xác).
  - Với `p(z) ∝ N(0, I)·exp(β⟨ω(z), e⟩)` thì `∂E[ω_k]/∂e_k = β·Var(ω_k) ≥ 0`.
  - DPI gốc (e = G, return dự đoán): QoS dự đoán giảm thì ưu tiên QoS giảm.
  - Bản sửa (e = mức thiếu hụt so với mục tiêu): QoS thiếu thì ưu tiên QoS tăng.
- **(d) P so với I.**
  - Bằng chứng là thiếu hụt hiện tại thì giống điều khiển P: dao động, còn sai lệch thường trực. Lagrange là điều khiển I.
  - Stooke, Achiam, Abbeel, "Responsive Safety in RL by PID Lagrangian Methods", ICML 2020, PMLR 119:9133–9143.
  - Đề xuất: bằng chứng là hàng đợi ảo cộng phần P, tức PID-Lagrangian có ước lượng bất định.
- **Phần khó chứng minh:**
  - Nhiều agent hợp tác có ràng buộc có thể có duality gap khác 0 (Kahe & Kebriaei, arXiv:2410.15335).
  - Môi trường đổi giữa episode.
  - Toàn bộ vòng lặp kín với mạng nơ-ron: chỉ chứng minh được dưới giả định "policy ε-tối ưu với mỗi ω".

### Công trình liên quan tới việc dịch ω theo ràng buộc

| Công trình | Giống | Khác |
|---|---|---|
| CCPO (Yao et al., NeurIPS 2023, arXiv:2310.03718) | Policy có điều kiện theo ngưỡng ràng buộc, đổi lúc chạy | Ngưỡng do người đặt, cố định cả episode; một agent |
| PDOA (arXiv:2409.09958) | Đổi bài toán có ràng buộc thành MORL | Offline, cần demonstration |
| Lyapunov-guided DRL (arXiv:2409.13580, 2501.15928, 2603.27364) | Hàng đợi ảo cho UAV và slicing | Hàng đợi nhúng vào reward lúc train |
| SafeSlicing (Liu, Choi, Han, ICNP 2021, arXiv:2110.04320) | Lagrange cho SLA trong slicing | Một mục tiêu; nhân tử học lúc train |
| Kahe & Kebriaei (arXiv:2410.15335) | Primal-dual phân tán, nhân tử riêng mỗi agent | Không có policy có điều kiện theo ω |

Phần còn mới (sau khoảng 10 lượt tìm kiếm, chưa phải rà soát đầy đủ; nên kiểm thêm bằng Elicit):
- ω_i dịch lúc chạy theo KPI của nhà mạng, có cơ sở đối ngẫu và hàng đợi ảo, không train lại;
- trên policy đa mục tiêu nhiều agent có prior đội hình;
- so sánh với DPI và với ω học được.

### Mỗi agent một preference

| Bài | ω_i từ đâu | Reward | Ghi chú |
|---|---|---|---|
| **PCMA**: Wang, Guo, Xie, Liu, Cao, Chen, "Learning Coordinated Preference for MOMARL", arXiv:2606.14693 (v2 9/2026) | Planner rút `p_i ~ Dir(φ_ψ(o_i))` | Reward chung cho đội cộng vector riêng mỗi agent: `R_i = r_team·1_K + r_i` | Mục tiêu đội cố định; xem chi tiết dưới |
| MIRL: Xia & Herrmann, arXiv:2610.35736 (10/2026) | Học bằng meta-gradient để phục vụ ω_team do người dùng đặt; MIRL-D đổi từng bước theo lịch sử cục bộ | **Vector chung cho cả đội** (Def. 2.2) | MO-COMA, liên tục; chênh với baseline nhỏ, nhiều chỗ trong một độ lệch chuẩn |
| AA-MAMORL: Li, Luo, Huang, Qi, Min, arXiv:2511.08926 (11/2025) | Cho sẵn hoặc `w_i = g_i(o_i)` | Vector | Mỗi agent có lợi ích riêng (cách 1); critic attention trên token (o_j, a_j, w_j); MAMO-DDPG; cân bằng Nash Bayes tồn tại khi các agent quan sát được ω của nhau |
| MAGPIE (Mu et al., arXiv:2608.08604, IEEE T-ASE) | **Không phải MORL** | — | "Preference" là phản hồi so sánh từng cặp, dùng để học reward vô hướng |

**PCMA chi tiết** (Sec. 1–5 đọc nguyên văn; thí nghiệm chỉ đọc qua bản tóm tắt):
- Lợi ích của agent i: `U_i = J_team + p_iᵀ J_i`. Advantage của actor: `A_i = A_team + λ2·p_iᵀ A_i^ind` (Eq. 1).
- Planner train bằng PPO trên advantage của đội trừ đi một thưởng đa dạng: `D_p = (1/2N²)ΣΣ‖p_i − p_j‖²` (Eq. 3).
- Critic đội và critic vector cho từng agent (Eq. 2). CTDE, PPO.
- **Định lý 4.2:** đa dạng preference cộng thêm `ηNκD_p` vào mức cải thiện của đội, nhưng chỉ khi có giả định căn chỉnh 4.1 (κ > 0). Thưởng đa dạng thì được cộng bất kể giả định có đúng không.
- **Định lý 4.6:** nếu `‖p^{t+1} − p^t‖ ≤ δ` thì sai lệch so với cân bằng bị chặn bởi `Cδ/(1−ρ)` (cần giả định co 4.4 và Lipschitz 4.5). Dùng được cho trường hợp ω đổi lúc chạy, và là lý do để giữ ω đổi mượt.
- **Thí nghiệm:** 3 seed; MPE, Catch, Escort, MOMAWalker, SMAC, OpenCDA-MARL (tối đa 10 agent). Các con số chưa đối chiếu.

**Ánh xạ PCMA sang bài UAV:**
- J_team ↔ mục tiêu của nhà mạng: tổng có trọng số theo ω_team, hoặc hàm Lagrange của các ràng buộc.
- J_i ↔ mục tiêu riêng của UAV i: QoS các UE nó phục vụ, năng lượng của nó, công bằng giữa các UE của nó. **Môi trường phải tách reward theo UAV.**
- **Môi trường đã có lý do cho vai trò khác nhau:** episode kết thúc khi **bất kỳ** UAV nào chạm năng lượng cần để quay về. Vì vậy UAV pin yếu tự ưu tiên năng lượng là có lợi cho cả đội.

### Kiến trúc đề xuất bản 2 (phân tầng; chưa vẽ lại sơ đồ)

1. **Tầng đội:** mục tiêu của nhà mạng, cập nhật PID-Lagrangian hoặc hàng đợi ảo trên KPI của cả mạng, cho ra ω_team lúc chạy. Có cơ sở toán (a), (b), (d).
2. **Tầng UAV:** planner kiểu PCMA, nhận (o_i, ω_team), cho ra p_i. Train để tối đa hoá lợi ích của đội; có thể thêm thưởng đa dạng.
3. **Actor attention:** token gồm bản thân, láng giềng (thêm **p_j của láng giềng**, theo AA-MAMORL), vùng nhu cầu, p_i. Cộng với prior theo nhu cầu.
4. **Ghi đè:** người vận hành đặt ω_team cho cả mạng, hoặc p_i cho một UAV.

Câu hỏi mở thêm:
- Tách reward theo UAV thế nào, nhất là công bằng.
- Planner train on-policy (PPO như PCMA) trong khi actor dùng TD3 off-policy, hay dùng meta-gradient kiểu MIRL?
- Đa dạng p_i có thật sự giúp không (giả định κ > 0 của PCMA)? Phải kiểm bằng thí nghiệm, so với p_i = ω_team cho mọi UAV.

### Các quyết định còn mở (sơ đồ bản 2, cùng link ở trên)

Các mục ghi "quyết trước" chặn việc sửa môi trường v2.

**Hỏi thầy:**
- **D1 Mục tiêu vận hành và tín hiệu** (quyết trước):
  - KPI nào (SLA từng slice, pin dự trữ, độ phủ) và ngưỡng bao nhiêu.
  - Viết thành bài toán "tối ưu X với điều kiện Y ≥ c".
- **D2 Mật độ người dùng** (quyết trước): nhà mạng có ước lượng được không? Có thì dùng prior theo nhu cầu; không thì giữ tam giác cố định, hoặc chỉ dùng số UE mà các UAV đang phục vụ.
- **D8 Ghi đè ở tầng nào:**
  - Có ba lựa chọn: ω_team cho cả mạng, p_i cho một UAV, hoặc cả hai.
  - Khi ghi đè p_i của một UAV, các UAV còn lại có tự bù không?
- **D9 Chia sẻ p_j giữa các UAV:** UAV có gửi vai trò cho nhau được không (giả định liên lạc)? Theo AA-MAMORL, cần quan sát được preference của nhau thì mới có điểm cân bằng.
- **D12 Mức lý thuyết:** định lý chứng minh đầy đủ (dưới giả định policy ε-tối ưu), hay mệnh đề kèm kiểm chứng bằng thí nghiệm.

**Mình đề xuất, cần bạn đồng ý:**
- **D3 Tách reward theo UAV** (quyết trước):
  - Đề xuất: theo PCMA, reward đội cộng vector riêng. Phần riêng gồm QoS của các UE mà UAV đó phục vụ và năng lượng của chính nó. Công bằng chỉ giữ ở mức đội.
  - Lựa chọn khác: theo MIRL, chỉ dùng vector của đội.
- **D10 Môi trường v2 cụ thể** (quyết trước):
  - Kịch bản nào: tan concert, tải URLLC tăng, gió mạnh.
  - N đổi giữa ca, hay chỉ khác N giữa các episode.
  - Sửa eMBB: scheduler chia lại RB thừa, hàng đợi riêng mỗi slice, ngưỡng 50 Mbps.
- **D7 TD3 hay thêm entropy:** entropy làm hàm đối ngẫu trơn, nên cơ sở toán của tầng đội vững hơn. Đổi lại phải đổi thuật toán gốc so với paper MOMA-AC.
- **D11 Baseline và quota:**
  - Bắt buộc: `ma3_prior` chạy lại trên v2, PCMA nguyên bản, mọi UAV dùng chung p_i = ω_team, DPI gốc.
  - Tuỳ chọn: MIRL, AA-MAMORL.

**Quyết bằng thí nghiệm nhỏ:**
- **D4 Cách train planner:** PPO ở nhịp chậm (PCMA), meta-gradient (MIRL), hoặc tối ưu qua critic như một hành động bậc cao. Cách cuối là ý của mình, chưa ai kiểm chứng.
- **D5 Thưởng đa dạng p_i:** có hay không, và hệ số λ1 bao nhiêu. Ablation so với p_i = ω_team cho mọi UAV.
- **D6 Nhịp cập nhật:** ω_team và p_i đổi mỗi bước hay mỗi khung vài bước; hệ số P, I, D; mức chặn δ cho mỗi lần đổi (PCMA Thm 4.6). Thử trên checkpoint có sẵn, chưa cần Kaggle.

**Thứ tự đề xuất:**
1. Chốt D1, D2, D3, D10 với thầy, rồi sửa môi trường v2.
2. Thử D6 trên checkpoint có sẵn.
3. Chốt D7 và D11, rồi cài planner, actor và critic attention.
4. Pilot D4 và D5, rồi train đầy đủ.
5. Bàn D8, D9, D12 song song.

## Link

- PCMA: https://arxiv.org/abs/2606.14693
- MIRL: https://arxiv.org/html/2610.35736
- AA-MAMORL: https://arxiv.org/html/2511.08926v1
- Toure et al.: https://arxiv.org/abs/2601.14092
- CAPQL: https://github.com/haoyelu/CAPQL
- Paternain et al.: https://arxiv.org/abs/1910.13393
- Ying et al.: https://proceedings.mlr.press/v151/ying22a.html
- PID Lagrangian: https://proceedings.mlr.press/v119/stooke20a.html
- CCPO: https://arxiv.org/abs/2310.03718
- SafeSlicing: https://arxiv.org/abs/2110.04320
- Kahe & Kebriaei: https://arxiv.org/abs/2410.15335
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
