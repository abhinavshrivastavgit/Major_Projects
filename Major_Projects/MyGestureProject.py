import cv2
from HandTrackingModule import HandDetector # You are importing YOUR own tool!

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print("Camera is captured!")
detector = HandDetector() # Initialize your "Machine"

while True:
    success, img = cap.read()
    
    # NEW SAFETY CHECK: Only process if the camera actually gave us an image
    if not success or img is None:
        print("Waiting for camera frame...")
        continue

    img = detector.find_hands(img) 
    lm_list = detector.get_position(img) 
    
    if len(lm_list) != 0:
        print(f"Index Finger is at: {lm_list[8][1]}, {lm_list[8][2]}")

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break