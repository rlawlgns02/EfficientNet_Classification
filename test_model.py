import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import argparse

def load_damage_detection_model(model_path):
    """모델 로드 함수"""
    try:
        model = load_model(model_path)
        print(f"모델을 '{model_path}'에서 성공적으로 로드했습니다.")
        return model
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        return None

def preprocess_image(img_path):
    """이미지 전처리 함수"""
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        processed_img = preprocess_input(img_array)
        return processed_img
    except Exception as e:
        print(f"이미지 전처리 중 오류 발생: {e}")
        return None

def predict_damage(model, img_array, class_names, grade_mapping):
    """파손 등급 예측 함수"""
    try:
        # 예측 수행
        predictions = model.predict(img_array)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = class_names[predicted_class_idx]
        confidence = predictions[0][predicted_class_idx] * 100
        
        # 등급 매핑
        predicted_grade = None
        for grade, classes in grade_mapping.items():
            if predicted_class in classes:
                predicted_grade = grade
                break
        
        return {
            'grade': predicted_grade,
            'class': predicted_class,
            'confidence': confidence,
            'predictions': predictions[0]
        }
    except Exception as e:
        print(f"예측 중 오류 발생: {e}")
        return None

def main():
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description='핸드폰 파손 감지 테스트')
    parser.add_argument('--image', type=str, required=True, help='테스트할 이미지 경로')
    parser.add_argument('--model', type=str, default='/home/ubuntu/phone_damage_detection/models/phone_damage_EfficientNetB0.h5', 
                        help='모델 파일 경로')
    args = parser.parse_args()
    
    # 클래스 이름 및 등급 매핑
    class_names = ['back_broken', 'back_normal', 'front_broken', 'front_lines', 'front_normal']
    grade_mapping = {
        'A': ['back_normal', 'front_normal'],  # 완전 새거처럼 멀정함
        'B': ['front_lines'],                  # 약간의 생활 기스가 있음
        'C': ['front_broken'],                 # 액정이 눈에 띄게 파손됨
        'F': ['back_broken']                   # 사용불가능할 정도로 파손됨
    }
    
    # 등급별 설명
    grade_descriptions = {
        'A': "완전 새거처럼 멀정한 상태입니다.",
        'B': "약간의 생활 기스가 있는 상태입니다.",
        'C': "액정이 눈에 띄게 파손된 상태입니다.",
        'F': "사용불가능할 정도로 파손된 상태입니다."
    }
    
    # 모델 로드
    model = load_damage_detection_model(args.model)
    if model is None:
        return
    
    # 이미지 전처리
    processed_img = preprocess_image(args.image)
    if processed_img is None:
        return
    
    # 예측 수행
    result = predict_damage(model, processed_img, class_names, grade_mapping)
    if result is None:
        return
    
    # 결과 출력
    print("\n===== 핸드폰 파손 감지 결과 =====")
    print(f"파일: {os.path.basename(args.image)}")
    print(f"파손 등급: {result['grade']}급")
    print(f"상세 설명: {grade_descriptions[result['grade']]}")
    print(f"감지된 상태: {result['class']}")
    print(f"신뢰도: {result['confidence']:.2f}%")
    
    # 각 클래스별 확률 출력
    print("\n각 클래스별 확률:")
    for i, class_name in enumerate(class_names):
        print(f"{class_name}: {result['predictions'][i]*100:.2f}%")
    
    # 이미지 표시
    plt.figure(figsize=(8, 8))
    img = image.load_img(args.image)
    plt.imshow(img)
    plt.title(f"파손 등급: {result['grade']}급 ({result['class']}, 신뢰도: {result['confidence']:.2f}%)")
    plt.axis('off')
    
    # 결과 이미지 저장
    result_dir = '/home/ubuntu/phone_damage_detection/results'
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, f"result_{os.path.basename(args.image)}")
    plt.savefig(result_path)
    print(f"\n결과 이미지가 '{result_path}'에 저장되었습니다.")
    
    print("\n참고: 현재 모델의 정확도는 제한적입니다. 결과를 참고용으로만 사용해주세요.")

if __name__ == "__main__":
    main()
