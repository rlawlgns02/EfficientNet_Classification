import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0, ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

# 데이터 경로 설정
data_dir = 'D:/prjAi2/phone_classification-2'
train_dir = os.path.join(data_dir, 'train')
valid_dir = os.path.join(data_dir, 'valid')
test_dir = os.path.join(data_dir, 'test')

# 이미지 크기 및 배치 크기 설정
img_height, img_width = 224, 224
batch_size = 32

# 데이터 증강 및 전처리를 위한 ImageDataGenerator 설정
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# 데이터 로드 및 전처리
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# 클래스 이름과 인덱스 매핑 확인
class_indices = train_generator.class_indices
print("클래스 인덱스:", class_indices)
num_classes = len(class_indices)
print(f"총 클래스 수: {num_classes}")

# 각 클래스별 이미지 수 확인
class_counts = {}
for class_name in os.listdir(train_dir):
    class_path = os.path.join(train_dir, class_name)
    if os.path.isdir(class_path):
        class_counts[class_name] = len(os.listdir(class_path))
print("각 클래스별 학습 이미지 수:", class_counts)

# 사용자 정의 등급과 원래 클래스 매핑
grade_mapping = {
    'A': ['back_normal', 'front_normal'],  # 완전 새거처럼 멀정함
    'B': ['front_lines'],                  # 약간의 생활 기스가 있음
    'C': ['front_broken'],                 # 액정이 눈에 띄게 파손됨
    'F': ['back_broken']                   # 사용불가능할 정도로 파손됨
}

# 매핑 정보 출력
print("\n사용자 정의 등급과 원래 클래스 매핑:")
for grade, classes in grade_mapping.items():
    print(f"{grade}급: {', '.join(classes)}")

# 샘플 이미지 시각화 함수
def plot_sample_images(generator, num_images=5):
    plt.figure(figsize=(15, 10))
    
    # 배치 가져오기
    images, labels = next(generator)
    
    # 클래스 이름 리스트
    class_names = list(generator.class_indices.keys())
    
    # 이미지 표시
    for i in range(min(num_images, len(images))):
        plt.subplot(1, num_images, i+1)
        plt.imshow(images[i])
        class_idx = np.argmax(labels[i])
        class_name = class_names[class_idx]
        
        # 등급 찾기
        grade = None
        for g, classes in grade_mapping.items():
            if class_name in classes:
                grade = g
                break
        
        plt.title(f"{class_name}\n({grade}급)")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('D:/prjAi2/sample_images.png')
    plt.close()

# 샘플 이미지 시각화
try:
    plot_sample_images(train_generator)
    print("샘플 이미지가 'D:/prjAi2/sample_images.png'에 저장되었습니다.")
except Exception as e:
    print(f"샘플 이미지 시각화 중 오류 발생: {e}")

print("\n데이터 전처리가 완료되었습니다.")
