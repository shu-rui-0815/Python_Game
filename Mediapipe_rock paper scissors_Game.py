import cv2
import mediapipe as mp
import random
import time

# 初始化模組
mp_hands = mp.solutions.hands   # 初始化 MediaPipe 的 Hand 模組
mp_drawing = mp.solutions.drawing_utils     # 畫出手部關鍵點

# 定義手勢
def get_hand_sign(landmarks ,hand_type):
    # 建立串列 fingers，記錄每隻手指是否為伸直
    fingers = []

    if hand_type == "Right":
        if landmarks[4].x < landmarks[3].x:
            fingers.append(1)
        else:
            fingers.append(0)
    elif hand_type == "Left":
        if landmarks[4].x > landmarks[3].x:  
            fingers.append(1)
        else:
            fingers.append(0)

    # 其他手指頭(透過y座標來判斷是否張開)
    finger_tips = [8, 12, 16, 20]
    finger_dips = [6, 10, 14, 18]

    for tip, dip in zip(finger_tips, finger_dips):
        if landmarks[tip].y < landmarks[dip].y:
            fingers.append(1)
        else:
            fingers.append(0)

    # 根據 fingers 來判斷手勢
    if fingers == [0, 0, 0, 0, 0]:
        return "rock"    
    elif fingers == [1, 1, 1, 1, 1]:
        return "paper"    
    elif fingers == [0, 1, 1, 0, 0]:
        return "scissors"
    else:
        return "unknown"  # 無法辨識


# 比對結果
def get_result(player, computer):
    if player == computer:
        return 'Draw'
    elif (player == 'rock' and computer == 'scissors') or (player == 'scissors' and computer == 'paper') or (player == 'paper' and computer == 'rock'):
        return 'You Win!'
    else:
        return 'You Lose!'


# 定義初始變數
cooldown = 3   # 每回合倒數時間（秒）
pause_time = 2    # 顯示結果後暫停時間（秒）
last_time = time.time() # 紀錄上回合開始的時間
paused = False # 控制目前是否進入暫停狀態
pause_start = 0
computer = "waiting"
player = "waiting"
result = "Waiting..."

# 開啟攝影機
cap = cv2.VideoCapture(0)

# 最多追蹤 1 隻手，偵測手部(低誤判率)，持續追蹤手部(低容錯率)
with mp_hands.Hands(
    max_num_hands=1,
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


        # 在偵測不到手的狀態， player 的狀態皆為 waiting
        player = "waiting"

        # 偵測手部
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_type = handedness.classification[0].label  # "Left" 或 "Right"
                # 畫出手部關鍵點與骨架
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                # 取得手勢
                player = get_hand_sign(hand_landmarks.landmark, hand_type)


        # 現在時間
        current_time = time.time()
        
        # 不是在暫停階段，記錄倒數時間
        if not paused:
            elapsed_time = current_time - last_time  # 計算時間差
            countdown = max(0, int(cooldown - elapsed_time))  # 計算倒數秒數
        else:
            countdown = 0 
                
        # 若未暫停且倒數超過三秒
        if not paused and elapsed_time >= cooldown:
            computer = random.choice(["rock", "paper", "scissors"])
            result = get_result(player, computer)
            paused = True
            pause_start = current_time

        # 若暫停且超過2秒
        if paused and current_time - pause_start >= pause_time:
            paused = False
            last_time = current_time
            result = "Waiting..."
            computer = "waiting"

        # 顯示倒數
        cv2.putText(image, f"Countdown: {countdown}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        # 顯示玩家的出拳
        cv2.putText(image, f"player: {player}", (425,450),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        # 顯示電腦的出拳
        cv2.putText(image, f'computer: {computer}', (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        # 顯示結果
        result_color = (0, 0, 255) if result == "You Win!" else (0, 0, 0)
        cv2.putText(image, f"Result: {result}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, result_color, 2)

        # 顯示畫面
        cv2.imshow('Rock Paper Scissors', image)

        # 按下 q 離開
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# 關閉攝影機與視窗
cap.release()
cv2.destroyAllWindows()
