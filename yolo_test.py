import cv2
import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression
from utils.general import xyxy2xywh, xywh2xyxy
from utils.torch_utils import select_device
from csv import writer
from datetime import datetime
coordinate = list()
with open('test_csv.csv', 'a', newline='') as f_object:
    writer_object = writer(f_object)
    writer_object.writerow(['Time','x1','x2','y1','y2'])
    def scale_coords(img_shape, coords, im0_shape):
        # img_shape : (w, h)
        # coords : [[x1, y1, x2, y2], ... ]
        # im0_shape : (h, w, c)

        gain = float(max(img_shape) / max(im0_shape))  # gain = old / new
        pad = (img_shape[1] - im0_shape[0] * gain) / 2, (img_shape[0] - im0_shape[1] * gain) / 2  # wh padding
        coords[:, [0, 2]] -= pad[0]  # x padding
        coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, :4] /= gain
        coords[:, :4] = torch.clamp(coords[:, :4], min=0)
        coords[:, [0, 2]] = torch.clamp(coords[:, [0, 2]], max=img_shape[0])
        coords[:, [1, 3]] = torch.clamp(coords[:, [1, 3]], max=img_shape[1])
        return coords


    # 모델 경로와 임계값, 비율 등 하이퍼파라미터를 설정합니다.
    weights_path = 'yolov5s.pt'
    conf_thres = 0.5
    iou_thres = 0.5
    device = select_device('cpu')

    # YOLOv5 모델을 로드합니다.
    model = attempt_load(weights_path, device=device)
    model.eval()

    # 웹캠으로부터 영상을 받아오는 객체를 생성합니다.
    cap = cv2.VideoCapture(0)

    while True:
        # 웹캠으로부터 영상 프레임을 가져옵니다.
        ret, frame = cap.read()

        if not ret:
            break

        # YOLOv5 모델에 입력할 이미지를 전처리합니다.
        img = frame.copy()  # 이미지 데이터를 복사합니다.
        img = img.transpose(2, 0, 1)  # 이미지 데이터의 메모리 레이아웃을 변경합니다.
        img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)

        # 모델을 이용하여 객체를 탐지합니다.
        detections = model(img)[0]
        detections = non_max_suppression(detections, conf_thres, iou_thres)

        # 탐지된 객체를 화면에 출력합니다.
        if detections is not None:
            detections = detections[0]
            detections[:, :4] = scale_coords(img.shape[2:], detections[:, :4], frame.shape).round()

            for *xyxy, conf, cls in detections:
                x1, y1, x2, y2 = [int(i) for i in xyxy]
                label = f"{model.names[int(cls)]} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                coordinate = map(str,[datetime.now(),x1,x2,y1,y2])
                # coordinate = list(map(str, coordinate))
                writer_object.writerow(coordinate)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # 화면에 출력된 영상을 보여줍니다.
        cv2.imshow('frame', frame)

        # 'q' 키를 누르면 종료합니다.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 사용한 자원을 해제합니다.
    f_object.close()
    cap.release()
    cv2.destroyAllWindows()


      
    
    