import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import tkinter as tk
from tkinter import filedialog, Label, Button, Frame, Text, END, DISABLED, Scrollbar
from PIL import Image, ImageTk
import cv2

class PhoneDamageDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("핸드폰 파손 감지 AI")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")
        
        # 모델 로드
        self.model_path = '/home/ubuntu/phone_damage_detection/models/phone_damage_EfficientNetB0.h5'
        self.model = load_model(self.model_path)
        
        # 클래스 이름 및 등급 매핑
        self.class_names = ['back_broken', 'back_normal', 'front_broken', 'front_lines', 'front_normal']
        self.grade_mapping = {
            'A': ['back_normal', 'front_normal'],  # 완전 새거처럼 멀정함
            'B': ['front_lines'],                  # 약간의 생활 기스가 있음
            'C': ['front_broken'],                 # 액정이 눈에 띄게 파손됨
            'F': ['back_broken']                   # 사용불가능할 정도로 파손됨
        }
        
        # 등급별 설명
        self.grade_descriptions = {
            'A': "완전 새거처럼 멀정한 상태입니다.",
            'B': "약간의 생활 기스가 있는 상태입니다.",
            'C': "액정이 눈에 띄게 파손된 상태입니다.",
            'F': "사용불가능할 정도로 파손된 상태입니다."
        }
        
        # UI 구성
        self.create_widgets()
        
    def create_widgets(self):
        # 상단 프레임 - 제목
        top_frame = Frame(self.root, bg="#4a7abc", height=60)
        top_frame.pack(fill="x")
        
        title_label = Label(top_frame, text="핸드폰 파손 감지 AI", font=("Arial", 18, "bold"), bg="#4a7abc", fg="white")
        title_label.pack(pady=10)
        
        # 중앙 프레임 - 이미지 및 결과
        center_frame = Frame(self.root, bg="#f0f0f0")
        center_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # 이미지 프레임
        self.image_frame = Frame(center_frame, bg="#ffffff", width=400, height=400, bd=2, relief="groove")
        self.image_frame.pack(side="left", padx=10, pady=10)
        self.image_frame.pack_propagate(False)
        
        self.image_label = Label(self.image_frame, bg="#ffffff", text="이미지가 여기에 표시됩니다")
        self.image_label.pack(fill="both", expand=True)
        
        # 결과 프레임
        result_frame = Frame(center_frame, bg="#ffffff", width=300, height=400, bd=2, relief="groove")
        result_frame.pack(side="right", padx=10, pady=10)
        result_frame.pack_propagate(False)
        
        result_title = Label(result_frame, text="분석 결과", font=("Arial", 14, "bold"), bg="#ffffff")
        result_title.pack(pady=10)
        
        # 결과 텍스트 영역
        self.result_text = Text(result_frame, height=15, width=35, wrap="word", font=("Arial", 10))
        self.result_text.pack(padx=10, pady=5, fill="both", expand=True)
        self.result_text.insert(END, "이미지를 업로드하면 분석 결과가 여기에 표시됩니다.")
        self.result_text.config(state=DISABLED)
        
        # 스크롤바
        scrollbar = Scrollbar(result_frame, command=self.result_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.result_text.config(yscrollcommand=scrollbar.set)
        
        # 하단 프레임 - 버튼
        bottom_frame = Frame(self.root, bg="#f0f0f0", height=100)
        bottom_frame.pack(fill="x", pady=10)
        
        upload_button = Button(bottom_frame, text="이미지 업로드", command=self.upload_image, 
                              font=("Arial", 12), bg="#4a7abc", fg="white", padx=20, pady=10)
        upload_button.pack(side="left", padx=20)
        
        predict_button = Button(bottom_frame, text="파손 감지", command=self.predict_damage, 
                               font=("Arial", 12), bg="#4CAF50", fg="white", padx=20, pady=10)
        predict_button.pack(side="right", padx=20)
        
        # 이미지 저장 변수
        self.uploaded_image = None
        self.processed_image = None
        
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="이미지 파일 선택",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        
        if file_path:
            # 이미지 로드 및 표시
            img = Image.open(file_path)
            img = img.resize((350, 350), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            
            self.image_label.config(image=photo)
            self.image_label.image = photo
            
            # 원본 이미지 저장
            self.uploaded_image = file_path
            
            # 결과 텍스트 초기화
            self.result_text.config(state="normal")
            self.result_text.delete(1.0, END)
            self.result_text.insert(END, f"이미지가 업로드되었습니다.\n파일: {os.path.basename(file_path)}\n\n'파손 감지' 버튼을 클릭하여 분석을 시작하세요.")
            self.result_text.config(state=DISABLED)
            
            # 전처리된 이미지 준비
            self.processed_image = self.preprocess_image(file_path)
    
    def preprocess_image(self, img_path):
        # 이미지 로드 및 전처리
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        processed_img = preprocess_input(img_array)
        return processed_img
    
    def predict_damage(self):
        if self.processed_image is None:
            self.result_text.config(state="normal")
            self.result_text.delete(1.0, END)
            self.result_text.insert(END, "먼저 이미지를 업로드해주세요.")
            self.result_text.config(state=DISABLED)
            return
        
        # 예측 수행
        predictions = self.model.predict(self.processed_image)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = self.class_names[predicted_class_idx]
        confidence = predictions[0][predicted_class_idx] * 100
        
        # 등급 매핑
        predicted_grade = None
        for grade, classes in self.grade_mapping.items():
            if predicted_class in classes:
                predicted_grade = grade
                break
        
        # 결과 표시
        self.result_text.config(state="normal")
        self.result_text.delete(1.0, END)
        self.result_text.insert(END, f"예측 결과:\n\n")
        self.result_text.insert(END, f"파손 등급: {predicted_grade}급\n\n")
        self.result_text.insert(END, f"상세 설명: {self.grade_descriptions[predicted_grade]}\n\n")
        self.result_text.insert(END, f"감지된 상태: {predicted_class}\n")
        self.result_text.insert(END, f"신뢰도: {confidence:.2f}%\n\n")
        
        self.result_text.insert(END, "참고: 현재 모델의 정확도는 제한적입니다. 결과를 참고용으로만 사용해주세요.")
        self.result_text.config(state=DISABLED)

# 메인 함수
def main():
    root = tk.Tk()
    app = PhoneDamageDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
