import cv2
import torch
import pandas as pd
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image

# Load Hugging Face model
processor = AutoImageProcessor.from_pretrained("SeanHowFun/face-emotion-detection")
model = AutoModelForImageClassification.from_pretrained("SeanHowFun/face-emotion-detection")

# Emotion labels
labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Video input/output
location_videofile = "./video_samples/video_sample_1.mp4"
cap = cv2.VideoCapture(location_videofile)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("./output/video_sample_1_output.mp4", fourcc,
                      cap.get(cv2.CAP_PROP_FPS),
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

# Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Accumulate emotion scores
emotion_totals = {label: 0.0 for label in labels}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_img = cv2.resize(face, (224, 224))
        face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))

        inputs = processor(images=face_pil, return_tensors="pt")
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].detach().numpy()

        # Update totals
        for i, label in enumerate(labels):
            emotion_totals[label] += probs[i]

        # Draw bounding box + top emotion
        top_idx = probs.argmax()
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
        cv2.putText(frame, f"{labels[top_idx]}: {probs[top_idx]*100:.2f}%",
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    out.write(frame)

cap.release()
out.release()

# Convert totals to DataFrame
emotions_df = pd.DataFrame({
    "Human Emotions": labels,
    "Emotion Value from the Video": [round(v, 2) for v in emotion_totals.values()]
})
print("Output:\n", emotions_df)
