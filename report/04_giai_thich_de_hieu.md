# Giải thích bằng lời thường (dùng khi trình bày)

## Bài toán trong một câu

Ba chiếc UAV làm "trạm phát wifi bay" cho một khu phố. Chúng phải tự quyết định bay đi đâu, phát mạnh hay yếu, và chia băng thông cho ai, sao cho cân bằng ba thứ ngược chiều nhau: **mạng tốt**, **tiết kiệm pin** và **công bằng giữa người dùng**. Người vận hành có thể đổi mức ưu tiên giữa ba thứ này bất cứ lúc nào, nên một bộ điều khiển phải làm tốt ở mọi mức ưu tiên.

## Phương pháp đề xuất trong một câu

Thay vì để AI học bay từ con số 0, ta cho nó một **"bài mẫu"** là đứng thành hình tam giác phủ đều khu phố. AI chỉ cần học cách **chỉnh sửa** bài mẫu đó, đồng thời tự học cách chia công suất và băng thông. Thêm vào đó, AI được dặn **ưu tiên hệ quả gần** (γ 0.9) thay vì cố tính quá xa (γ 0.99).

**Phép so sánh:** học lái xe có người chỉ đường sẵn rồi tự điều chỉnh, so với tự mò đường trong thành phố lạ.

## Kết quả chính

- **Tốt nhất:** điểm tổng hợp (hypervolume) 0.332, so với 0.176 của phương pháp trong paper gốc và 0.274 của các cách điều khiển thủ công tốt nhất.
- **Rất ổn định:** 5 lần train cho điểm gần như giống hệt nhau (0.330–0.333). Phương pháp gốc thì có lần được 0.25, có lần chỉ 0.026.
- **Cần cả hai thành phần:** chỉ có bài mẫu, hoặc chỉ đổi γ, thì chỉ ngang cách thủ công. Có cả hai mới vượt hẳn.

## Vì sao UAV "bay tuần" thay vì đứng yên

Với loại UAV này, **lơ lửng một chỗ tốn pin hơn bay chậm**, giống trực thăng phải gồng hơn khi treo trên không. AI tự phát hiện ra điều đó: mỗi UAV bay đi bay lại quanh vị trí của mình, như bảo vệ đi tuần quanh cổng. Nhờ vậy vẫn phủ sóng tốt mà tiêu thụ chỉ khoảng một phần ba năng lượng so với cách đứng yên ở tam giác.

Ngược lại, lần train hỏng của phương pháp gốc cho UAV **đứng im ở mép bản đồ**, nên gần như không phục vụ được ai.

## Vì sao người xem video gần như không ai đạt yêu cầu

- **Đòi hỏi quá cao:** mỗi người xem video cần 50 Mbps, nhưng hơn nửa số người này ở trong nhà, nơi sóng từ UAV khó xuyên qua tường. Người bắt được sóng chỉ đạt khoảng 3–4 Mbps.
- **Không phải lỗi của phương pháp:** dồn toàn bộ băng thông cho họ thì cũng chỉ khoảng 5% đạt. Mọi phương pháp, kể cả cách thủ công, đều chỉ được 1–2%.
- **So sánh vẫn công bằng:** họ vẫn được tính điểm một phần (4/50 Mbps vẫn có điểm).
- **Đây là hạn chế của kịch bản,** sẽ được hiệu chỉnh trong khoá luận.

## Câu hỏi thầy cô có thể hỏi và gợi ý trả lời

| Câu hỏi | Gợi ý trả lời |
|---|---|
| Sao chọn γ 0.9? | Theo tầm nhìn của bài toán (tác động của một hành động tắt sau khoảng 2 bước, episode dài 18 bước) và một pilot thô. Chưa tune có hệ thống; việc quét γ có kế hoạch cho khoá luận. Ablation cũng cho baseline dùng γ 0.9 để so công bằng. |
| Có phải chỉ nhờ heuristic tốt? | Không. Heuristic tốt nhất được 0.274, còn đề xuất được 0.332, và AI học được cách bay, chia băng thông và công suất mà heuristic không có. |
| Kết quả có "học tủ" không? | Số cuối đo trên seed test 200000+, chưa từng dùng khi phát triển phương pháp; 5 seed; khoảng tin cậy tính trên cả seed lẫn episode. |
| Nếu thêm hoặc bớt UAV thì sao? | Hiện tại mới cố định 3 UAV. Số UAV thay đổi là hướng của khoá luận. |
