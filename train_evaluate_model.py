import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

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
    class_mode='categorical',
    shuffle=True
)

validation_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# 클래스 이름과 인덱스 매핑 확인
class_indices = train_generator.class_indices
class_names = list(class_indices.keys())
print("클래스 인덱스:", class_indices)
num_classes = len(class_indices)
print(f"총 클래스 수: {num_classes}")

# 사용자 정의 등급과 원래 클래스 매핑
grade_mapping = {
    'A': ['back_normal', 'front_normal'],  # 완전 새거처럼 멀정함
    'B': ['front_lines'],                  # 약간의 생활 기스가 있음
    'C': ['front_broken'],                 # 액정이 눈에 띄게 파손됨
    'F': ['back_broken']                   # 사용불가능할 정도로 파손됨
}

# 모델 생성 함수
def create_model(num_classes=5):
    # EfficientNetB0 모델 로드 (ImageNet 가중치 사용)
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    
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
    
    return model

# 모델 생성
model = create_model(num_classes=num_classes)
print("EfficientNetB0 모델이 생성되었습니다.")

# 모델 저장 경로
model_dir = 'D:/prjAi2/models'
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, 'phone_damage_EfficientNetB0.h5')

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

# 학습 파라미터 설정
epochs = 30
steps_per_epoch = len(train_generator)
validation_steps = len(validation_generator)

print(f"학습 시작: 에폭 수={epochs}, 배치 크기={batch_size}")
print(f"학습 데이터 스텝 수={steps_per_epoch}, 검증 데이터 스텝 수={validation_steps}")

# 모델 학습
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=callbacks,
    verbose=1
)

# 학습 결과 시각화
def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    
    # 정확도 그래프
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # 손실 그래프
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('D:/prjAi2/training_history.png')
    plt.close()

# 학습 결과 시각화
plot_training_history(history)
print("학습 결과가 'D:/prjAi2/training_history.png'에 저장되었습니다.")

# 최적 모델 로드
best_model = load_model(model_path)
print(f"최적 모델을 '{model_path}'에서 로드했습니다.")

# 테스트 데이터로 모델 평가
test_loss, test_accuracy = best_model.evaluate(test_generator)
print(f"테스트 정확도: {test_accuracy:.4f}, 테스트 손실: {test_loss:.4f}")

# 예측 및 분류 보고서 생성
def evaluate_model(model, generator):
    # 예측
    generator.reset()
    y_pred_probs = model.predict(generator, steps=len(generator))
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # 실제 레이블
    y_true = generator.classes
    
    # 분류 보고서
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # 혼동 행렬
    cm = confusion_matrix(y_true, y_pred)
    
    return report, cm, y_true, y_pred

# 테스트 데이터로 모델 평가
report, cm, y_true, y_pred = evaluate_model(best_model, test_generator)

# 분류 보고서 출력
print("\n분류 보고서:")
for class_name in class_names:
    precision = report[class_name]['precision']
    recall = report[class_name]['recall']
    f1_score = report[class_name]['f1-score']
    support = report[class_name]['support']
    print(f"{class_name}: 정밀도={precision:.4f}, 재현율={recall:.4f}, F1 점수={f1_score:.4f}, 지원 수={support}")

print(f"\n전체 정확도: {report['accuracy']:.4f}")

# 혼동 행렬 시각화
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig('D:/prjAi2/confusion_matrix.png')
plt.close()
print("혼동 행렬이 'D:/prjAi2/confusion_matrix.png'에 저장되었습니다.")

# 사용자 정의 등급 매핑에 따른 성능 평가
def evaluate_by_grade(y_true, y_pred, class_names, grade_mapping):
    # 클래스 인덱스 매핑 생성
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
    
    # 등급별 정확도 계산
    grade_results = {}
    
    for grade, classes in grade_mapping.items():
        # 해당 등급에 속하는 클래스의 인덱스 찾기
        grade_indices = [i for i, true_class in enumerate(y_true) if class_names[true_class] in classes]
        
        if grade_indices:
            # 해당 등급의 샘플에 대한 예측 정확도 계산
            correct = sum(1 for i in grade_indices if class_names[y_pred[i]] in classes)
            accuracy = correct / len(grade_indices)
            grade_results[grade] = {
                'accuracy': accuracy,
                'samples': len(grade_indices)
            }
        else:
            grade_results[grade] = {
                'accuracy': 0,
                'samples': 0
            }
    
    return grade_results

# 사용자 정의 등급별 성능 평가
grade_results = evaluate_by_grade(y_true, y_pred, class_names, grade_mapping)

# 등급별 성능 출력
print("\n사용자 정의 등급별 성능:")
for grade, result in grade_results.items():
    print(f"{grade}급: 정확도={result['accuracy']:.4f}, 샘플 수={result['samples']}")

print("\n모델 학습 및 평가가 완료되었습니다.")
