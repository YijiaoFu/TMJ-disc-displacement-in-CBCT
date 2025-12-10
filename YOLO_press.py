
from ultralytics import YOLO
import cv2
import os
import numpy as np
import pandas as pd
import SimpleITK as sitk


original = pd.read_excel('./dataset/CBCT/patient_info.xlsx') #记录患者TMD情况的表格，由[patient_ID,label]两列组成
dict = original.set_index(original.columns[0])[original.columns[1]].to_dict()

def detect(img_file):
    det_model = YOLO("./mode_pth/YOLO_tmd/best.pt")
    matrix_expanded = np.expand_dims(img_file, axis=-1)
    img0 = np.repeat(matrix_expanded, 3, axis=-1)
    results = det_model.predict(img0, show=False)
    boxes = results[0].boxes.xyxy.tolist()
    return boxes

# 分别处理训练集和测试集
type_list = ['train','test']
class_type = ['class0','class1']
for type in type_list:
    # YOLO识别的TMJ切片存放目录
    crops_folder = './dataset/TMD_yolo_pre'
    for cla in class_type:
        # 患者CBCT存放目录，TMD目录下分为class0（正常）和class1（异常）两个子目录，每个子目录又按7:3划分为训练集和测试集两个子目录train和test;
        patient_path = './dataset/TMD/%s/%s'%(cla,type)
        patient_list = os.listdir(patient_path) # 目录名称为患者ID
        series_reader = sitk.ImageSeriesReader()
        wincenter = 150
        winwidth = 1500
        mini = int(wincenter - winwidth / 2.0)
        maxi = int(wincenter + winwidth / 2.0)
        intensityWindow = sitk.IntensityWindowingImageFilter()
        intensityWindow.SetWindowMaximum(maxi)
        intensityWindow.SetWindowMinimum(mini)
        for patient in patient_list:
            patient_label = dict[patient]
            if patient_label == '正常':
                crops_folder = os.path.join(crops_folder,'%s/class0'%(type))
            elif patient_label == '异常':
                crops_folder = os.path.join(crops_folder,'%s/class1'%(type))
            else:
                continue
            file_path = os.path.join(patient_path,patient)
            file_count = len(os.listdir(file_path))
            if file_count < 30 and file_count>0:
                subfolders = [os.path.join(file_path, f) for f in os.listdir(file_path) if os.path.isdir(os.path.join(file_path,f))]
                if len(subfolders)>0:
                    file_path = subfolders[0]
                else:
                    continue
            if not os.listdir(file_path):   
                os.rmdir(file_path)
            elif file_count > 100:
                dicom_names = series_reader.GetGDCMSeriesFileNames(file_path)
                series_reader.SetFileNames(dicom_names)
                image = series_reader.Execute()
                sitkImage = intensityWindow.Execute(image)
                trImage = sitk.GetArrayFromImage(sitkImage)
                z = trImage.shape[0]
                y = trImage.shape[1]
                x = trImage.shape[2]
                for i in range(0,x): # 切片id
                    img = trImage[:,:,i]
                    img = cv2.flip(img,-1)
                    img = img.astype(np.uint8)
                    boxes = detect(img)
                    if len(boxes) > 0:
                        boxes = boxes[0]
                        crop_img = img[int(boxes[1]):int(boxes[3]), int(boxes[0]):int(boxes[2])]
                        save_path = os.path.join(crops_folder, f"{patient}_{i}.jpg")
                        cv2.imwrite(save_path, crop_img)
                crops_folder = './dataset/TMD_yolo_pre'