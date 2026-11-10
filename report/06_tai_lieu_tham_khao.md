# Tài liệu tham khảo (kèm phần cụ thể đã dùng)

Danh sách đầy đủ các nguồn của môi trường, kèm phương trình và bảng đã dùng, nằm trong tài liệu kế hoạch (Claude Doc: https://claude.ai/artifact/3XSWT3vUcLBZv4SyrZ2js2) và trong comment của `config/environment/ma3.yaml`. Dưới đây là các nguồn cần trích trong báo cáo.

**Cần kiểm tra lại trước khi nộp:** các mục đánh dấu (\*) được ghi theo trí nhớ hoặc chưa đối chiếu trực tiếp với bản gốc. Phải kiểm tra số trang, phương trình và năm.

## Phương pháp học

- MOMA-AC (paper gốc của baseline), Alg. 1 (MOMA-TD3), Eq. 1–2, 6, 10–12. Lấy citation đầy đủ từ tài liệu kế hoạch.
- Fujimoto, van Hoof, Meger. "Addressing Function Approximation Error in Actor-Critic Methods" (TD3). ICML 2018. Dùng Sec. 5.1: sai lệch ước lượng tích luỹ theo γ.
- Silver, Allen, Tenenbaum, Kaelbling. "Residual Policy Learning". arXiv:1812.06298, 2018. Dùng Sec. IV.
- Johannink et al. "Residual Reinforcement Learning for Robot Control". ICRA 2019, arXiv:1812.03201. Dùng Eq. 5.
- Andrychowicz et al. "What Matters in On-Policy Reinforcement Learning? A Large-Scale Empirical Study". ICLR 2021, arXiv:2006.05990. Dùng Sec. 3.6: γ là một trong các siêu tham số quan trọng nhất, nên tune theo môi trường; tập quét ở App. I.1.
- Eimer, Lindauer, Raileanu. "Hyperparameters in Reinforcement Learning and How To Tune Them". ICML 2023, arXiv:2306.01324. Dùng Sec. 6 (seed tuning/test riêng, ngân sách tuning bằng nhau) và App. A (≥ 5 seed).
- Agarwal et al. "Deep Reinforcement Learning at the Edge of the Statistical Precipice". NeurIPS 2021. Dùng stratified bootstrap và probability of improvement (Sec. 5).
- (\*) Jiang, Kulesza, Singh, Lewis. "The Dependence of Effective Planning Horizon on Model Accuracy". AAMAS 2015. Ý chính: tầm nhìn ngắn có tác dụng điều chuẩn. Định lý chưa đối chiếu.
- Jain, Chiu, Hawe. "A Quantitative Measure of Fairness and Discrimination for Resource Allocation in Shared Computer Systems". DEC-TR-301, 1984. Dùng Sec. 5 mục 6: tỉ lệ được phục vụ.

## Mô hình môi trường

- Zeng, Xu, Zhang. "Energy Minimization for Wireless Communication With Rotary-Wing UAV". IEEE TWC 2019, bản arXiv. Dùng mô hình năng lượng bay, Table I.
- Dai et al. Mô hình năng lượng UAV có gió, arXiv:2304.06909. Dùng Eq. 8.
- Al-Hourani, Kandeepan, Lardner. "Optimal LAP Altitude for Maximum Coverage". IEEE WCL 2014. Dùng P_LoS và tham số dense urban.
- Galkin et al. Backhaul mmWave cho UAV, arXiv:1710.10807. Dùng Eq. 3.
- Lee et al. Mô hình log-normal tương quan không gian của mật độ tải. IEEE Wireless Commun. 21(1), 2014. Dùng Eq. 1–2 và Table 1 (urban DL).
- Mogensen et al. Mô hình Shannon có suy giảm (SNR_eff 1.25).
- EARTH project. Mô hình công suất micro BS: 112 + 2.6 P_tx.
- ITU-R P.1410 (lưới nhà); 3GPP TR 38.901 (Table 7.4.3-3 suy hao xuyên tường, Table 7.5-6 khoảng tương quan, mẫu anten); TR 36.777 (shadowing UMa-AV); TS 22.261 (Table 7.1-1, Table 7.2.3.2-1); TR 38.913 Sec. 7.6; TR 45.820; TR 36.885 (lưới đường); TR 38.864 Table B-1 (NF).
- DJI M300 RTK User Manual v1.8, tr. 22 (Low Battery RTH).
- Global Wind Atlas (Hà Nội); hoa gió trạm Láng.
