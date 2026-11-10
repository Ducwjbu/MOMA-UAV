# Hạn chế (ghi vào báo cáo) và hướng cho khoá luận

| # | Hạn chế | Bằng chứng | Hướng xử lý (khoá luận) |
|---|---|---|---|
| 1 | γ 0.9 được chọn trước theo tầm nhìn và một pilot thô, không tune với ngân sách như nhau cho mọi phương pháp | CLAUDE.md, mục formation prior | Quét γ ∈ {0.5, 0.8, 0.9, 0.95, 0.99} (đều theo log tầm nhìn) trên seed validation, cùng ngân sách cho mọi phương pháp (Andrychowicz et al. 2021 Sec. 3.6; Eimer et al. 2023 Sec. 6) |
| 2 | SLA eMBB chỉ đạt 1–2% với mọi policy | 03_ket_qua.md mục 8: thông lượng là chỉ số hỏng; 56% ở trong nhà; cận trên 4.9% | Hiệu chỉnh mục tiêu eMBB hoặc tải; xem mục 3 và 4 bên dưới |
| 3 | RB cấp cho vùng nhu cầu không có UE hoạt động bị bỏ phí (scheduler không chia lại) | `_allocate_rbs_fairly` trong `environment.py` | Scheduler chia lại RB thừa (work-conserving) |
| 4 | Mỗi UAV có một hàng đợi chung cho 3 slice, nên UAV quá tải thì delay của mọi slice đều tăng (không cách ly slice) | Trung vị delay hàng đợi eMBB 99 ms ở heuristic | Hàng đợi riêng cho từng slice |
| 5 | Mới có một kịch bản: 3 UAV, khu 2 km, bản đồ tải log-normal | — | Số UAV thay đổi; kịch bản ngoài phân bố huấn luyện (tan concert, điểm nóng di chuyển) |
| 6 | Vòng lặp kín rất nhạy: CPU, GPU và Kaggle cho HV hơi khác nhau | CLAUDE.md, mục Evaluation | Luôn báo cáo số từ `eval_checkpoint.py` trên CPU, với 20 episode |
| 7 | Tam giác đứng yên được đặt sẵn tại chỗ, không tốn chi phí bay ra | `eval_formation.py` | Chỉ là tham chiếu; họ heuristic xuất phát ngẫu nhiên như agent |
| 8 | Thực thi chỉ phân tán ở mức mạng: quan sát của UAV i gồm vị trí và công suất của UAV khác | CLAUDE.md, Known pitfalls | Giả định có liên kết giữa các UAV; nêu rõ trong báo cáo |
| 9 | Đội hình prior cố định (tam giác quanh tâm), không theo nhu cầu | `prior.py` | Prior theo nhu cầu (weighted k-means), gán lại đỉnh bằng thuật toán Hungarian khi số UAV đổi |
| 10 | Một số chi tiết của MOMA-TD3 theo TD3 chứ không theo đúng phương trình của paper (done masking, clamp sau khi cộng nhiễu) | CLAUDE.md, Known pitfalls | Nêu trong phần cài đặt |
