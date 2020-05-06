import numpy as np
import os.path
import cv2
import matplotlib.pyplot as plt
from align import AlignDlib
from model import create_model
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC


nn4_small2_pretrained = create_model()
nn4_small2_pretrained.load_weights('models/nn4.small2.v1.h5')


class IdentityMetadata():
    def __init__(self, base, name, file):
        self.base = base
        self.name = name
        self.file = file

    def __repr__(self):
        return self.image_path()

    def image_path(self):
        return os.path.join(self.base, self.name, self.file)


def load_metadata(path):
    metadata = []
    for i in os.listdir(path):
        for f in os.listdir(os.path.join(path, i)):
            ext = os.path.splitext(f)[1]
            if ext == '.jpg' or ext == '.jpeg':
                metadata.append(IdentityMetadata(path, i, f))

    np.save('metadata', metadata)
    return np.array(metadata)

def load_image(path):
    img = cv2.imread(path, 1)
    return img[..., ::-1]

def align_image(img):
    return alignment.align(96, img, alignment.getLargestFaceBoundingBox(img),
                           landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)


metadata = load_metadata('images')

# Initialize the OpenFace face alignment utility
alignment = AlignDlib('models/landmarks.dat')


# 這裡會創建 128 是因為預先訓練的模型，架構是 96*96的 input 的全連接層，輸出128個點為特徵值
#創建零的陣列然後依照image 裡面有幾張圖就創幾個，每一個都是128個零
embedded = np.zeros((metadata.shape[0], 128))
print(" 0: ", embedded.shape)


for i, m in enumerate(metadata):
    #這個是把圖片讀到 img ，因為metadata 有把圖片的位置另外記錄下來，
    #所以這裡可以用讀圖的 fuction 搭配 metadata 的圖片位置

    img = load_image(m.image_path())

    img = align_image(img)

    # 把0-255的RGB縮放到區間[0,1]
    img = (img / 255.).astype(np.float32)

    #輸入是96*96 ，但是透過預先的model 去提取128個特徵
    # Keras要求第一維度是batch，所以要expand_dim把img從(96,96,3)變成(1,96,96,3)
    embedded[i] = nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]

def distance(emb1, emb2):
    return np.sum(np.square(emb1 - emb2))

####################a4
distances = []  # squared L2 distance between pairs
identical = []  # 1 if same identity, 0 otherwise

num = len(metadata)

for i in range(num - 1):
    for j in range(1, num):
        print("i J :",i ,j )
        print("distance : ",distance(embedded[i], embedded[j]))
        distances.append(distance(embedded[i], embedded[j]))
        identical.append(1 if metadata[i].name == metadata[j].name else 0)

distances = np.array(distances)
identical = np.array(identical)

thresholds = np.arange(0.3, 1.5, 0.01)

f1_scores = [f1_score(identical, distances < t) for t in thresholds]
acc_scores = [accuracy_score(identical, distances < t) for t in thresholds]

opt_idx = np.argmax(f1_scores)
# Threshold at maximal F1 score
opt_tau = thresholds[opt_idx]
# Accuracy at maximal F1 score
opt_acc = accuracy_score(identical, distances < opt_tau)

# Plot F1 score and accuracy as function of distance threshold
plt.plot(thresholds, f1_scores, label='F1 score')
plt.plot(thresholds, acc_scores, label='Accuracy')
plt.axvline(x=opt_tau, linestyle='--', lw=1, c='lightgrey', label='Threshold')
# plt.title('Accuracy at threshold {opt_tau:.2f} = {opt_acc:.3f}')
plt.title('Accuracy at threshold {opt_tau:.2f} = {opt_acc:.3f}'.format(opt_tau=opt_tau,opt_acc=opt_acc))
# plt.title('Accuracy at threshold { opt_tau}'opt_tau +'='+ str(opt_acc))
print(opt_tau)
plt.xlabel('Distance threshold')
plt.legend()
plt.show()



targets = np.array([m.name for m in metadata])
print(targets)

encoder = LabelEncoder()
encoder.fit(targets)

y = encoder.transform(targets)
print(y)
print("------------")

train_idx = np.arange(metadata.shape[0]) % 1 == 0
test_idx = np.arange(metadata.shape[0]) % 1 == 0

X_train = embedded[train_idx]
X_test = embedded[test_idx]

y_train = y[train_idx]
y_test = y[test_idx]

knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
svc = LinearSVC()

knn.fit(X_train, y_train)
svc.fit(X_train, y_train)


from sklearn.externals import joblib


joblib.dump(svc, 'models/svc.pkl')

print(svc.predict(X_test))
print("----")

acc_knn = accuracy_score(y_test, knn.predict(X_test))
acc_svc = accuracy_score(y_test, svc.predict(X_test))

print('KNN accuracy = ',acc_knn,' SVM accuracy = ',acc_svc)

