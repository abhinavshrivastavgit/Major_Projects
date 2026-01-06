import cv2
import mediapipe as mp
import time

# 1. Initialize System Architecture [cite: 2026-01-04]
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# 2. Start Video Capture
cap = cv2.VideoCapture(0)

print("Starting Camera... Press 'q' to exit.")

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    # Flip for natural 'mirror' feel and convert color
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 3. Technical Logic: Process Hand Landmarks [cite: 2026-01-04]
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            # Draw the 'skeletal' connections
            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)
            
            # Data Literacy: Accessing specific coordinates [cite: 2026-01-04]
            # ID 8 is the Index Finger Tip
            for id, lm in enumerate(hand_lms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                if id == 8: # Focus on Index Finger
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

    # Display result
    cv2.imshow("AI-Gesture-HCI Engine", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()