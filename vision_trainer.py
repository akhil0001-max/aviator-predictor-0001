import cv2
import pytesseract

# (Windows ke liye yeh path set karna padta hai, agar error aaye to uncomment kar lena)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def start_camera_training():
    cap = cv2.VideoCapture(0)
    print("Training started... Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray)
        print("Detected:", text)

        cv2.imshow("Camera Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\n[INFO] Collected multipliers:")
    for t, m in multiplier_history:
        print(f"{t} - {m}x")
