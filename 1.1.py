import datetime
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import probs as probs
import seaborn as sns
import tensorflow as tf
from sklearn.utils import class_weight
from tensorflow.python.keras.backend import dropout

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import general_header as gh
from sklearn.preprocessing import LabelBinarizer

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.applications import VGG19
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Dense, MaxPool2D, Flatten
from tensorflow.keras.models import Sequential
from IPython.display import set_matplotlib_formats
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import roc_curve
from xgboost import XGBClassifier
from yellowbrick.classifier import ROCAUC
from imblearn.over_sampling import SMOTE

set_matplotlib_formats('retina')
now = datetime.datetime.now()
runned = now.strftime("%Y-%m-%d %H:%M:%S")
print(runned + " : 구동")

result_path = '/data/source/POC/plot/' + runned + '/'
os.mkdir(result_path)

# sys.stdout = open(result_path + 'console.txt', 'w')

#photo_path = '/data/workspace/POCdata/photo/ENG'  # 진짜
# photo_path = '/data/workspace/POCdata/photo/1/ENG' #학명만
# photo_path = '/data/workspace/POCdata/photo/Legacy/test' #테스트
#photo_path = '/data/workspace/POCdata/photo/poc'  # poc
photo_path = '/data/workspace/POCdata/photo/inat2021'  #inat

photo_dir = os.listdir(photo_path)

'''########## 로그 파일 생성 ##########'''

dev = [4, 5, 6]
gh.set_gpu(dev)

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    '''#################################################################### 포맷 및 이미지 크기 설정 #####################################################################'''

    labels = photo_dir
    img_size = 224


    def get_data(data_dir):
        data = []
        for label in tqdm(labels, desc='get_data'):
            path = os.path.join(data_dir, label)
            class_num = labels.index(label)
            for img in os.listdir(path):
                try:
                    img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
                    # bgr_arr = cv2.split(img_arr)
                    # rgb_arr = cv2.merge([bgr_arr[2], bgr_arr[1], bgr_arr[0]])  # BGR -> RGB 코드
                    # resized_arr = cv2.resize(rgb_arr, (img_size, img_size))
                    # del img_arr  # 이미지 크기 변경하기(RGB로)
                    resized_arr = cv2.resize(img_arr, (img_size, img_size))  # 이미지 크기 변경하기
                    data.append([resized_arr, class_num])
                except Exception as e:
                    print(e)
        np.save('pocdata.npy', data)
        return np.array(data, dtype="object")



    #data = get_data('/data/workspace/POCdata/photo/ENG') # 진짜
    # data = get_data('/data/workspace/POCdata/photo/1/ENG')  # 학명만
    # data = get_data('/data/workspace/POCdata/photo/Legacy/test')  # 테스트
    #data = get_data('/data/workspace/POCdata/photo/poc')  # poc
    data = get_data('/data/workspace/POCdata/photo/inat2021')  # 정제한거

    print("다 넣음")
    # data = data.shuffle(buffer_size=len(data)) # 시간 잡아먹어서 끔
    #data = np.load('pocdata.npy', allow_pickle=True)

    '''#################################################################### 총 입력 이미지 수 시각화 #####################################################################'''

    l = []
    for i in data:
        l.append(labels[i[1]])
    sns.set(rc={'figure.figsize': (76.8, 43.2)})
    sns.set_style('dark')
    sns.countplot(l)

    plt.xticks(rotation=90, fontsize=25)
    plt.yticks(rotation=90, fontsize=90)

    plt.savefig(result_path + 'totalamount.png')
    print("TotalImage 출력...")

    '''#################################################################### 배열에 이미지 넣기 #####################################################################'''

    x = []
    y = []

    print("x,y 배열 삽입")
    for feature, label in tqdm(data):
        x.append(feature)
        y.append(label)
    '''#################################################################### 증강 #####################################################################'''

    '''#################################################################### 그레이스케일 정규화 및 리쉐이핑 #####################################################################'''

    x = np.array(x) / 255  # 데이터 노멀라이징
    x = x.reshape(-1, img_size, img_size, 3)  # 3차원 배열화
    y = np.array(y)  # 라벨값

    label_binarizer = LabelBinarizer()
    y = label_binarizer.fit_transform(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=0)

    '''#################################################################### ROC커브 #####################################################################'''


    def plot_roc_curve(fper, tper):
        plt.plot(fper, tper, color='red', label='ROC')
        plt.plot([0, 1], [0, 1], color='green', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic Curve')
        plt.legend()
        plt.show()


    del x, y, data

    print("그레이스케일 완료")

    '''#################################################################### 모델 정의 #####################################################################'''

    print("WORKERS: ", strategy.num_replicas_in_sync)

    pre_trained_model = VGG19(input_shape=(224, 224, 3), include_top=False,
                              weights="imagenet")  # vgg19를 사용한 이유는 imagenet으로 pre-trained된 모델이기 때문에

    for layer in pre_trained_model.layers[:19]:
        layer.trainable = False
    num_classes = 30
    model = Sequential([
        pre_trained_model,
        MaxPool2D((2, 2), strides=2),
        Flatten(),
        Dense(num_classes, activation=tf.nn.softmax)])
    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    print("Model Summary 완료")
    '''#################################################################### 에포크 #####################################################################'''

    # learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.3,
    #                                             min_lr=0.000001)  # 콜백함수
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=1, verbose=1, mode='auto')
    history = model.fit(x_train, y_train, batch_size=64, epochs=5, validation_data=(x_test, y_test),
                        callbacks=[learning_rate_reduction])
    #model.save("1.1softmodel.h5")

    print("테스트에 사용할 이미지 수 :", len(y_test))

    '''#################################################################### 정확도  플롯 #####################################################################'''

    epochs = [i for i in range(5)]
    fig, ax = plt.subplots(1, 2)
    train_acc = history.history['accuracy']
    train_loss = history.history['loss']
    val_acc = history.history['val_accuracy']
    val_loss = history.history['val_loss']
    fig.set_size_inches(20, 10)

    ax[0].plot(epochs, train_acc, 'go-', label='Training Accuracy')
    ax[0].plot(epochs, val_acc, 'ro-', label='Validation Accuracy')
    ax[0].set_title('Training & Validation Accuracy')
    ax[0].legend()
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Accuracy")

    ax[1].plot(epochs, train_loss, 'g-o', label='Training Loss')
    ax[1].plot(epochs, val_loss, 'r-o', label='Validation Loss')
    ax[1].set_title('Training & Validation Loss')
    ax[1].legend()
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Loss")

    plt.savefig(result_path + 'accuracy & loss.png')

    predictions = model.predict_classes(x_test)
    print(predictions)
    y_test_inv = label_binarizer.inverse_transform(y_test)  # y_test의 인덱스를 라벨로 뱉어냄

    print(classification_report(y_test_inv, predictions, target_names=labels))
    print("모델 전체 정확도 : ", accuracy_score(y_test_inv, predictions))

    cm = confusion_matrix(y_test_inv, predictions)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    plt.figure(figsize=(100, 100))
    sns.heatmap(cm / np.sum(cm), cmap="Blues", linecolor='black', linewidth=1, annot=True, fmt='.1%',
                xticklabels=labels, yticklabels=labels)
    plt.savefig(result_path + 'avg.png')
    print("avg 출력")

    '''#################################################################### 예측한 결과값 저장 #####################################################################'''

    prop_class = []
    mis_class = []

    '''#################################################################### 예측과 실제가 일치하는 경우 #####################################################################'''

    i = 0
    for i in range(len(y_test_inv)):
        if (y_test_inv[i] == predictions[i]):
            prop_class.append(i)
        if (len(prop_class) == 8):
            break

    count = 0
    fig, ax = plt.subplots(4, 2)
    fig.set_size_inches(15, 15)
    for i in range(4):
        for j in range(2):
            ax[i, j].imshow(x_test[prop_class[count]])
            ax[i, j].set_title(
                "Predicted Flower : " + labels[predictions[prop_class[count]]] + "\n" + "Actual Flower : " + labels[
                    y_test_inv[prop_class[count]]])
            plt.tight_layout()
            count += 1

    plt.savefig(result_path + 'correctflowers.png')
    print("CorrectFlower 출력")
    '''#################################################################### 예측과 실제가 일치하지 않는 경우 #####################################################################'''

    i = 0
    for i in range(len(y_test_inv)):
        if (y_test_inv[i] != predictions[i]):
            mis_class.append(i)
        if (len(mis_class) == 8):
            break

    count = 0
    fig, ax = plt.subplots(4, 2)
    fig.set_size_inches(15, 15)
    for i in range(4):
        for j in range(2):
            ax[i, j].imshow(x_test[mis_class[count]])
            ax[i, j].set_title(
                "Predicted Flower : " + labels[predictions[mis_class[count]]] + "\n" + "Actual Flower : " + labels[
                    y_test_inv[mis_class[count]]])
            plt.tight_layout()
            count += 1

    plt.savefig(result_path + 'wrongflowers.png')
    print("wrongFlower 출력")

    '''#################################################################### 테스트 #####################################################################
    # 데이터 증강 및 드롭아웃 레이어는 추론 시 비활성화됨.
    
    
    test_path = '/data/workspace/POCdata/photo/test/'
    x = input("이름 입력 : ")
    example = test_path + x
    testimg = tf.keras.preprocessing.image.load_img(example, target_size=(img_size, img_size))
    
    img_array = tf.keras.preprocessing.image.img_to_array(testimg)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    
    testpredictions = model.predict(img_array)
    print(testpredictions)
    score = tf.nn.softmax(testpredictions[0])
    print(score)
    print("이 이미지는 {}(으)로 예측되고, 확률은 약 {:.2f}% 입니다.".format(labels[np.argmax(score)], 100 * np.max(score)))'''

    '''#################################################################### 루프 #####################################################################
    
    while True:
        testimg()
        if input("다시 테스트하시겠습니까? (y/n)") == "n":
            break
    
    '''

    # xgb_basic = XGBClassifier()
    # xgb_basic.fit(x_train, y_test_inv)
    #
    # visualizer = ROCAUC(xgb_basic, classes=[0, 1], micro=False, macro=True, per_class=False)
    # visualizer.fit(x_train, y_test_inv)
    # visualizer.score(x_train,y_test_inv)
    # visualizer.show()
    '''#################################################################### 저장 #####################################################################'''#################################################################### 종료 #####################################################################'''

    # sys.stdout.close()
    # logging.basicConfig(filename=result_path + '.log', level=logging.DEBUG)
    sys.exit(0)

