import os
import numpy as np
import cv2
from align import AlignDlib
from model import create_model
from sklearn.preprocessing import LabelEncoder
import csv
import shutil


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
        for f in os.listdir(os.path.join(path + '/' + i)):
            ext = os.path.splitext(f)[1]
            if ext == '.jpg' or ext == '.jpeg':
                metadata.append(IdentityMetadata(path, i, f))
                # print(metadata)                      
    metadata_np=np.array(metadata)     
    return metadata_np

def load_metadata_predict(path): #辨識時多張的情形，採取先進先出
    metadata = []
    for i in os.listdir(path):   
        DirFileName= os.listdir(os.path.join(path + '/' + i))
     
        for f in os.listdir(os.path.join(path + '/' + i)):         
            if f==DirFileName[0] :
                    ext = os.path.splitext(f)[1]
                    if ext == '.jpg' or ext == '.jpeg':             
                        metadata.append(IdentityMetadata(path, i, f))                              
    metadata_np=np.array(metadata)     
    return metadata_np

def normalization(metadata):  
    def load_image(path):
        img = cv2.imread(path, 1)
        return img[..., ::-1]
        
    def align_image(img):
        return alignment.align(96, img, alignment.getLargestFaceBoundingBox(img),
                               landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

    alignment = AlignDlib('models/landmarks.dat')  
    embedded = np.zeros((metadata.shape[0], 128))
    
    for i, m in enumerate(metadata):
        # noinspection PyBroadException
       try:
            img = load_image(m.image_path())
            img = align_image(img)
            img = (img / 255.).astype(np.float32)
            embedded[i] = nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]
            recognize_flag = 1
            
            del img,metadata
            
       except Exception as e:
            recognize_flag = 0
    return recognize_flag, embedded,i
    
def label(metadata):
    targets = np.array([m.name for m in metadata])
    encoder = LabelEncoder()
    encoder.fit(targets)
    return encoder

def distance(emb1, emb2):
    return np.sum(np.square(emb1 - emb2))

def recordToCSV(Photo_name,example_identity):
    with open('./output.csv', 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        NewName = Photo_name.rstrip('.jpg')
        writer.writerow([NewName, example_identity])

def move(movepath1,movepath2, Photo_name) :
    movepath1 = os.path.join(movepath1 + '/' + str(Photo_name))
    movepath2 = os.path.join(movepath2 + '/' + str(Photo_name))
    shutil.move(movepath1, movepath2)  # 辨識後移除照片
    print("move finish")
    print("-----------")

def copyimg(path_recognize_folder,Photo_name,path_finish2):
    img_copy = cv2.imread(path_recognize_folder + '/' + Photo_name[0])
    shapelist = img_copy.shape

    emptyImage = img_copy.copy()
    pic = cv2.resize(emptyImage, (shapelist[0], shapelist[1]), interpolation=cv2.INTER_CUBIC)

    if os.path.isfile(path_recognize_folder + '/' + Photo_name[0]):
        move(path_recognize_folder, path_finish2, Photo_name[0])
        cv2.imwrite(path_recognize_folder + '/' + Photo_name[0], pic)
    else:
        cv2.imwrite(path_recognize_folder + '/' + Photo_name[0], pic)
    return 0
    










