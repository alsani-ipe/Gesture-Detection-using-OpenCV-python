import cv2
import mediapipe as mp

webcam= cv2.VideoCapture(0)
hand= mp.solutions.hands
hands_drawings= mp.solutions.drawing_utils
with hand.Hands(min_detection_confidence= 0.5, min_tracking_confidence= 0.5) as hands:
    while True:
        control, frame= webcam.read()
        rgb= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result= hands.process(rgb)
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                hands_drawings.draw_landmarks(frame, hand_landmarks, hand.HAND_CONNECTIONS)
        cv2.imshow("Test", frame)
        if cv2.waitKey(10)==27:
            break
            
            