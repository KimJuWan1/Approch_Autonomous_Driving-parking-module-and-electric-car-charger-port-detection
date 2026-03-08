import sys
import cv2
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer, Qt
from ultralytics import YOLO

class ObjectDetection(QWidget): # gui 클래스
    def __init__(self):
        super().__init__()
        self.setWindowTitle('특수 효과 (우측 정렬)')
        
        # [수정] 창 지오메트리 조정 (640 + 320 + 여백)
        self.setGeometry(200, 200, 1000, 550) 

        # ------------------------------
        # 1. 레이아웃 및 라벨 설정
        # ------------------------------
        
        # 1-1. 왼쪽 원본 라벨
        self.label_original = QLabel(self)
        self.label_original.setFixedSize(640, 480)
        self.label_original.setStyleSheet("background-color: lightgray;")
        self.label_original.setAlignment(Qt.AlignCenter)

        # 1-2. 오른쪽 객체 영역 라벨
        self.label_cropped_object = QLabel(self)
        self.label_cropped_object.setFixedSize(320, 320) 
        self.label_cropped_object.setStyleSheet("background-color: #dddddd;")
        self.label_cropped_object.setAlignment(Qt.AlignCenter)
        self.label_cropped_object.setText("객체 감지 대기 중...")

        # 1-3. 오른쪽 객체 이름 라벨
        self.label_object_class = QLabel(self)
        self.label_object_class.setFixedWidth(320)     # 너비는 320으로 고정
        
        # 텍스트가 길어지면 자동 줄 바꿈 활성화
        self.label_object_class.setWordWrap(True)      
        
        # 1줄일 때도 최소 높이를 50으로 유지 (레이아웃이 떨리지 않게)
        self.label_object_class.setMinimumHeight(50)   

        self.label_object_class.setStyleSheet(
            "background-color: #cccccc; color: black; font-size: 18px; font-weight: bold;"
            "padding: 5px;" # 텍스트가 너무 붙지 않게 여백 추가
        ) 
        self.label_object_class.setAlignment(Qt.AlignCenter) # 중앙으로 배치
        self.label_object_class.setText("")

        quitButton=QPushButton('나가기',self) # 나가기 버튼 생성
        quitButton.setGeometry(640,10,100,30) 
        quitButton.clicked.connect(self.quitFunction)

        # 1-4. 오른쪽 수직 레이아웃 (변경 없음)
        vbox_right = QVBoxLayout()
        vbox_right.addWidget(self.label_cropped_object)
        vbox_right.addWidget(self.label_object_class)
        vbox_right.addWidget(quitButton)
        vbox_right.addStretch(1) # 하단 여백

        # 1-5. 전체 수평 레이아웃
        hbox_main = QHBoxLayout()
        hbox_main.addWidget(self.label_original) # 왼쪽 위젯 추가
        hbox_main.addStretch(1) 
        hbox_main.addLayout(vbox_right) # 오른쪽 레이아웃 추가

        # 1-6. 모든 것을 위로 밀어 올릴 메인 수직 레이아웃
        main_vbox = QVBoxLayout()
        main_vbox.addLayout(hbox_main) # 기존 수평 레이아웃을 넣고
        main_vbox.addStretch(1)       # 그 아래에 스트레치를 추가
        
        self.setLayout(main_vbox)     # [수정] 최종 레이아웃을 main_vbox로 설정



        # ------------------------------
        # 2. YOLO 모델 및 카메라 설정
        # ------------------------------
        try:
            self.model = YOLO("Y:\home\Drive\capstone_design_2025\코드\port_detection\\best.pt") # 사용자 모델 경로
            self.class_names = self.model.names
            print("YOLO 모델 로드 성공:", self.class_names)
        except Exception as e:
            QMessageBox.critical(self, "오류", f"YOLO 모델 로드 실패: {e}")
            return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "오류", "카메라를 열 수 없습니다.")
            return

        self.timer = QTimer()
        self.timer.timeout.connect(self.updateFrame)
        self.timer.start(30)

    # ------------------------------
    # 3. updateFrame
    # ------------------------------
    def updateFrame(self):
        ret, frame = self.cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            return

        results = self.model(frame, verbose=False)

        # --- 왼쪽: 원본 + 바운딩 박스 표시 ---
        annotated_frame_bgr = results[0].plot()
        annotated_frame_rgb = cv2.cvtColor(annotated_frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, c = annotated_frame_rgb.shape
        qimg_annotated = QImage(annotated_frame_rgb.data, w, h, w * c, QImage.Format_RGB888)
        scaled_pix_annotated = QPixmap.fromImage(qimg_annotated).scaled(
            self.label_original.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.label_original.setPixmap(scaled_pix_annotated)

        # --- 오른쪽: 잘라낸 객체 영역 처리 ---
        if len(results[0].boxes) > 0:
            box = results[0].boxes[0]
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy

            cls_index = int(box.cls[0])
            class_name = self.class_names[cls_index]
            confidence = float(box.conf[0])

            cropped_img_bgr = frame[y1:y2, x1:x2].copy()
            cropped_img_rgb = cv2.cvtColor(cropped_img_bgr, cv2.COLOR_BGR2RGB)

            if cropped_img_rgb.size > 0:
                h_c, w_c, c_c = cropped_img_rgb.shape
                qimg_cropped = QImage(cropped_img_rgb.data, w_c, h_c, w_c * c_c, QImage.Format_RGB888)
                
                # 라벨 크기가 320x320으로 줄었기 때문에 scaled 메서드로 크기에 맞게 비율 유지하며 스케일링함
                scaled_pix_cropped = QPixmap.fromImage(qimg_cropped).scaled(
                    self.label_cropped_object.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                self.label_cropped_object.setPixmap(scaled_pix_cropped)
                text_to_display = f"<font color='red'>{class_name}</font>가 감지되었습니다. 충전기를 연결해주십시오."
                self.label_object_class.setText(text_to_display) # 신뢰도 (f"({confidence*100:.1f}%)")
            else:
                self.label_cropped_object.clear()
                self.label_object_class.setText("객체 영역 오류")

        else:
            self.label_cropped_object.clear()
            self.label_cropped_object.setText("객체 감지 대기 중...")
            self.label_object_class.setText("")


    def quitFunction(self): # 종료 기능
        self.cap.release() # 카메라 연결 해제
        self.close() # pyqt 창 닫기

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ObjectDetection()
    win.show()
    sys.exit(app.exec_())