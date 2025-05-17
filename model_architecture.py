import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0, ResNet50
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

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

# 사용자 정의 등급과 원래 클래스 매핑
grade_mapping = {
    'A': ['back_normal', 'front_normal'],  # 완전 새거처럼 멀정함
    'B': ['front_lines'],                  # 약간의 생활 기스가 있음
    'C': ['front_broken' ], # 액정이 눈에 띄게 파손됨
    'F': ['back_broken', 'front_broken']   # 사용불가능할 정도로 파손됨
}

# 모델 선택 함수 (ResNet50 또는 EfficientNetB0)
def create_model(model_type='efficientnet', num_classes=5):
    if model_type.lower() == 'resnet50':
        # ResNet50 모델 로드 (ImageNet 가중치 사용)
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
        model_name = 'ResNet50'
    else:
        # EfficientNetB0 모델 로드 (ImageNet 가중치 사용)
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
        model_name = 'EfficientNetB0'
    
    # 기본 모델 동결 (가중치 고정)
    base_model.trainable = False
    
    # 새로운 분류 레이어 추가
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    # 모델 컴파일
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"{model_name} 모델이 성공적으로 생성되었습니다.")
    return model, model_name

# EfficientNetB0 모델 생성 (더 효율적인 모델)
model, model_name = create_model(model_type='efficientnet', num_classes=num_classes)

# 모델 요약 정보 출력
model.summary()

# 모델 저장 경로
model_dir = 'D:/prjAi2/models'
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, f'phone_damage_{model_name}.h5')

# 콜백 함수 설정
callbacks = [
    ModelCheckpoint(
        filepath=model_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
]

print(f"\n모델 아키텍처 구현이 완료되었습니다. 모델 타입: {model_name}")
print(f"다음 단계는 모델 학습 및 평가입니다.")
