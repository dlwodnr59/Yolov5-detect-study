YOLOv5 객체 검출 코드 - 학습용으로 수정했습니다.

소개
이 코드는 웹캠으로부터 영상을 받아와 YOLOv5 모델을 이용하여 객체 검출을 수행하는 코드입니다. 검출된 객체의 좌표를 csv 파일에 기록합니다.

사용 방법
best.pt 파일을 준비합니다. best.pt는 YOLOv5 모델의 가중치 파일로, YOLOv5 깃허브 페이지에서 다운로드할 수 있습니다.
cv2, torch, csv, datetime 패키지를 설치합니다.
test_csv.csv 파일을 생성합니다.
코드를 실행합니다.
기타 정보
conf_thres와 iou_thres는 모델의 하이퍼파라미터입니다.
device는 모델을 실행할 디바이스를 설정합니다. 현재는 CPU만을 사용합니다.
객체 검출 후 좌표를 저장할 csv 파일의 이름과 포맷을 설정합니다.
코드 실행 중 'q'를 누르면 종료됩니다.



실행 결과
![image](https://user-images.githubusercontent.com/47483492/229721633-3d01a453-399e-4388-a6f0-004e1a24610a.png)
