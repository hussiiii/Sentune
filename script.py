import cv2
from deepface import DeepFace
import pyautogui

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

last_emotion = None  # Variable to store the last detected emotion
emotion_duration = 0  # Counter to track duration of the current emotion
required_duration = 11  # Number of consecutive frames the emotion must be present

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Detect emotions on the current frame
    try:
        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        if analysis and isinstance(analysis, list) and len(analysis) > 0:
            current_emotion = analysis[0]['dominant_emotion']
        else:
            current_emotion = None
    except Exception as e:
        print("Error in emotion detection:", e)
        print("Analysis result:", analysis)
        current_emotion = None

    # Check if the current emotion is the same as the last detected emotion
    if current_emotion == last_emotion:
        emotion_duration += 1  # Increment the duration counter
    else:
        emotion_duration = 0  # Reset the counter if the emotion changes

    # Control Spotify based on the detected emotion
    if emotion_duration == required_duration:  # Only act when the emotion has been stable for required duration
        if current_emotion == "happy":
            pyautogui.press('space')  # Play/Pause
        elif current_emotion == "sad":
            pyautogui.keyDown('ctrl')  # Press down the control key
            pyautogui.press('left')  # Press the left arrow key
            pyautogui.keyUp('ctrl')  # Release the control key
        elif current_emotion == "surprise":
            pyautogui.keyDown('ctrl')  # Press down the control key
            pyautogui.press('right')  # Press the right arrow key
            pyautogui.keyUp('ctrl')  # Release the control key

    last_emotion = current_emotion  # Update the last detected emotion

    # Display the detected emotion and duration on the frame
    cv2.putText(frame, f"Emotion: {current_emotion} ({emotion_duration})", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and destroy all windows when done
cap.release()
cv2.destroyAllWindows()
