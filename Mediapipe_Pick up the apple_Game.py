import cv2
import mediapipe as mp
import pygame
import sys
import random

# 初始化 MediaPipe 模組
mp_hands = mp.solutions.hands   # 初始化 MediaPipe 的 Hand 模組
mp_drawing = mp.solutions.drawing_utils     # 畫出手部關鍵點

# 初始化 Pygame
pygame.init()
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 36)
# 創建視窗
WIDTH, HEIGHT = 640, 480
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("接蘋果吧！")


# 籃子設定
basket_img = pygame.Surface((100, 20)) # 寬100，長20的長方形
basket_img.fill((100, 200, 100)) # 綠色
# 設定起始位置
basket_x = WIDTH / 2
basket_y = HEIGHT - 50


# 蘋果設定
apple_x = random.randint(0, WIDTH - 30) # 隨機生成位置
apple_y = -20
apple_speed = 5 

# 設定初始分數
score = 0

# 開啟攝影機
cap = cv2.VideoCapture(0)

# 最多追蹤 2 隻手，偵測手部(低誤判率)，持續追蹤手部(低容錯率)
with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5) as hands:

    # 攝影機開啟
    while cap.isOpened():
        success, image = cap.read()
        # 若無法成功
        if not success:
            print("Cannot open camera.")
            break

        # 影像處理(影像翻轉，顏色轉換，將 openCV 用的 BGR 轉換成 MediaPipe 使用的 RGB)
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False # 將影像設為唯讀，提升效能
        results = hands.process(image) 
        # 將顏色由 RGB 轉回來 BGR，以利 OpenCV 顯示
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)   

        # 判斷手勢方向
        move_direction = "stop"

        # 確認是否有手部關節點的座標與左右手的資訊
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_label = handedness.classification[0].label  # "Left" or "Right"
                
                # 判斷是否握拳(透過食指~小指的指尖點大於或小於第二關節點來判斷)
                finger_status = []
                for tip_id in [8, 12, 16, 20]:  # 食指~小指
                    tip_y = hand_landmarks.landmark[tip_id].y
                    mcp_y = hand_landmarks.landmark[tip_id - 2].y  # 對應的 MCP
                    if tip_y > mcp_y:
                        finger_status.append(True) # 握拳
                    else:
                        finger_status.append(False) # 伸直
                
                if all(finger_status):
                    move_direction = "stop"
                else:
                    if hand_label == "Right":
                        move_direction = "right"
                    elif hand_label == "Left":
                        move_direction = "left"

                # 畫出手部關鍵點
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # 根據手勢移動籃子
        if move_direction == "left":
            basket_x -= 10
        elif move_direction == "right":
            basket_x += 10
        elif move_direction == "stop":
            pass  # 停止不動

        # 防止籃子移動超出畫面
        basket_x = max(0, min(WIDTH - 100, basket_x))

        # 更新蘋果位置
        apple_y += apple_speed
        if apple_y > HEIGHT:
            apple_y = -20
            apple_x = random.randint(0, WIDTH - 30)

        # 判斷是否接到蘋果
        if (basket_y < apple_y + 30 < basket_y + 20) and (basket_x < apple_x + 30 and apple_x < basket_x + 100):
            score += 1
            apple_y = -20
            apple_x = random.randint(0, WIDTH - 30)

        # 設定 screen
        screen.fill((200, 230, 255))  
        pygame.draw.circle(screen, (255, 0, 0), (apple_x, apple_y), 15) # 蘋果(即時顯示蘋果)
        screen.blit(basket_img, (basket_x, basket_y))  # 籃子
        score_text = font.render(f"Score: {score}", True, (0, 0, 0)) # 顯示分數
        screen.blit(score_text, (10, 10))

        pygame.display.flip()  # 更新畫面
        clock.tick(30)  # 限制每秒 30 幀

        # 顯示攝影機畫面 
        cv2.imshow('Hand Tracking', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

        # 關閉視窗
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                pygame.quit()
                sys.exit()

# 關閉攝影機與視窗
cap.release()
cv2.destroyAllWindows()
