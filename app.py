import cv2
import torch
import argparse

parser = argparse.ArgumentParser(description='Person detection in video')
parser.add_argument('--input', type=str, default='0', help='Path to input video file or camera ID')
parser.add_argument('--output', type=str, default='', help='Path to save output video file')
args = parser.parse_args()

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

if args.input.isdigit():
    cap = cv2.VideoCapture(int(args.input))
else:
    cap = cv2.VideoCapture(args.input)

output_writer = None
if args.output:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    results = model(frame)

    person_count = 0

    for *box, conf, cls in results.xyxy[0].cpu().numpy():
        if int(cls) == 0: 
            person_count += 1
            (startX, startY, endX, endY) = box
            cv2.rectangle(frame, (int(startX), int(startY)), (int(endX), int(endY)), (255, 255, 255), 2)
            label = f"Person: {conf:.2f}"
            cv2.putText(frame, label, (int(startX), int(startY) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.putText(frame, f"People Count: {person_count}", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow("Frame", frame)
    
    if output_writer:
        output_writer.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if output_writer:
    output_writer.release()