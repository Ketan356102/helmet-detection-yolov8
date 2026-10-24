import cv2
from ultralytics import YOLO
from datetime import datetime
import smtplib
from email.message import EmailMessage
import ssl
import os
import time
import threading

model = YOLO(r"C:\Users\ketan\helmet-detection-yolov8\runs\detect\train\weights\best.pt")

def send_alert_email(reciever_email, sender_email, sender_password, image_path):
    def send_email():
        try:
            subject = "⚠️ Helmet Violation Detected"
            body = "A person without a helmet has been detected. See attached image"

            msg = EmailMessage()
            msg['From'] = sender_email
            msg['To'] = reciver_email
            msg['Subject'] = subject
            msg.set_content(body)

            with open(image_path, 'rb') as f:
                img_data = f.read()
                msg.add_attachment(img_data, maintype='image', subtype='jpeg', filename=os.path.basename(image_path))

                context = ssl.create_default_context()
                with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
                    smtp.login(sender_email, sender_password)
                    smtp.send_message(msg)
                    print("Email sent successfully!")
        except Exception as e:
            print(f"Error sending email: {e}")

    email_thread = threading.Thread(target=send_email)
    email_thread.start()


sender_email = "23amtics367@gmail.com"
reciver_email = "23amtics367@gmail.com"
sender_password ="icuu vvgn ciyt lylx"


cap = cv2.VideoCapture(0)
last_alert_time = 0
alert_cooldown_seconds = 60  # Cooldown period in seconds

image_folder = r"C:\Users\ketan\helmet-detection-yolov8\no_helmet_images"
os.makedirs(image_folder, exist_ok=True)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to grab frame")
            break

        current_time = time.time()
        
        # Perform detection
        results = model(frame, stream=True)

        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf < 0.5:  # filter low-confidence detections
                    continue

                cls_id = int(box.cls[0])
                label = model.names[cls_id].lower()

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = (255, 255, 0)
                text = label

                if label == "with_helmet":
                    color = (0, 255, 0)
                    text = "With Helmet"

                elif label in ["without_helmet", "no_helmet"]:
                    color = (0, 0, 255)
                    text = "No Helmet"

                    # Non-blocking cooldown check
                    if current_time - last_alert_time >= alert_cooldown_seconds:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        image_path = os.path.join(image_folder, f"no_helmet_{timestamp}.jpg")
                        cv2.imwrite(image_path, frame)

                        # This will now run in a separate thread
                        send_alert_email(reciver_email, sender_email, sender_password, image_path)
                        last_alert_time = current_time
                        print("⏳ Alert sent, cooldown started...")

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Display real-time feed
        cv2.imshow("Helmet Detection", frame)
        
        # Check for quit command
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("🚀 Program terminated cleanly")


