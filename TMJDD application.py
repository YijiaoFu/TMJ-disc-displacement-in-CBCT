package main
import streamlit as st
import os
import torch
from torchvision import transforms
from PIL import Image
from timm import create_model
import SimpleITK as sitk
import numpy as np
import cv2
from ultralytics import YOLO
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, help='Input directory containing DCM files')
args = parser.parse_args()

custom_css = """
<style>
    div.stButton > button:first-child {
        background-color: #4CAF50;  # 设置背景颜色为绿色
        color: white;  # 设置文字颜色为白色
    }
</style>
"""


def predict(image_file):
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model('fastvit_t8', pretrained=False, num_classes=2)  #
    checkpoint = torch.load('temp/fvit_t8_med_classifier.pth', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载并处理图像
    image = image_file.convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    # 预测
    with torch.no_grad():
        outputs = model(image)
        _, predicted = outputs.max(1)

    return predicted.item()

def detect(img_file):
    det_model = YOLO("weights\best.pt")  
    matrix_expanded = np.expand_dims(img_file, axis=-1)  
    img0 = np.repeat(matrix_expanded, 3, axis=-1)  
    results = det_model.predict(img0, show=False)
    boxes = results[0].boxes.xyxy.tolist()
    return boxes


st.set_page_config("AI tool", ("jyfy.ico"), initial_sidebar_state="expanded")
st.title("Risk Assessment for TMJ disorders")
st.markdown(custom_css, unsafe_allow_html=True)

if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

input_dir = args.input_dir if args.input_dir else st.text_input("Enter DCM folder path")
if input_dir and os.path.exists(input_dir):
    # 递归查找所有DCM文件
    dcm_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.dcm'):
                dcm_files.append(os.path.join(root, file)) 
    if len(dcm_files) == 0:
        st.error("DCM file not found in the specified path")
    else:
        st.markdown(
            f'<div style="background-color: lightgrey; padding: 10px;">File Info: A total of {len(dcm_files)} DCM files</div>',
            unsafe_allow_html=True)
        if not st.session_state.analysis_complete and st.button("Upload and Analyze"):
            series_reader = sitk.ImageSeriesReader()
            dicom_names = series_reader.GetGDCMSeriesFileNames(input_dir)
            series_reader.SetFileNames(dicom_names)
            image = series_reader.Execute()
            wincenter = 150
            winwidth = 1500
            mini = int(wincenter - winwidth / 2.0)
            maxi = int(wincenter + winwidth / 2.0)
            intensityWindow = sitk.IntensityWindowingImageFilter()
            intensityWindow.SetWindowMaximum(maxi)
            intensityWindow.SetWindowMinimum(mini)
            sitkImage = intensityWindow.Execute(image)
            trImage = sitk.GetArrayFromImage(sitkImage)
            z = trImage.shape[0]
            y = trImage.shape[1]
            x = trImage.shape[2]
            if z >= 550 and x >= 700:  # 576 768 768
                low_sliceL = 200
                hig_sliceL = 250
                low_sliceR = 520
                hig_sliceR = 540
            if z >= 600 and x >= 600:  # 676 676 676
                low_sliceL = 200
                hig_sliceL = 280
                low_sliceR = 460
                hig_sliceR = 510
            elif z >= 500 and x > 600:  # 528 640 640
                low_sliceL = 90
                hig_sliceL = 140
                low_sliceR = 500
                hig_sliceR = 580
            elif z >= 550 and x > 550:  # 575 575 575
                low_sliceL = 150
                hig_sliceL = 200
                low_sliceR = 410
                hig_sliceR = 480
            elif z >= 450 and x > 600:  # 492 640 640
                low_sliceL = 90
                hig_sliceL = 110
                low_sliceR = 490
                hig_sliceR = 510
            elif z >= 400 and x > 500:  # 440 536 536
                low_sliceL = 50
                hig_sliceL = 100
                low_sliceR = 380
                hig_sliceR = 430
            elif z >= 200 and x > 600:  # 292 640 640
                low_sliceL = 90
                hig_sliceL = 110
                low_sliceR = 490
                hig_sliceR = 510
            elif z >= 300 and x > 300:  # 362 399 399
                low_sliceL = 90
                hig_sliceL = 100
                low_sliceR = 290
                hig_sliceR = 310
            predictions = []
            for i in range(low_sliceL, hig_sliceL):  # 切片id
                img = trImage[:,:,i]  # 获取第i个切片
                img = cv2.flip(img, -1) 
                img = img.astype(np.uint8)
                boxes = detect(img)
                if len(boxes) > 0:
                    boxes = boxes[0]
                    img_PIL = Image.fromarray(img[int(boxes[1]):int(boxes[3]), int(boxes[0]):int(boxes[2])])
                    output = predict(img_PIL)
                    predictions.append(output)
                else:
                    predictions.append(2)
            total_slices_L = len(predictions)
            for i in range(low_sliceR, hig_sliceR):  # 切片id
                img = trImage[:,:,i]  
                img = cv2.flip(img, -1) 
                img = img.astype(np.uint8)
                boxes = detect(img)
                if len(boxes) > 0:
                    boxes = boxes[0]
                    img_PIL = Image.fromarray(img[int(boxes[1]):int(boxes[3]), int(boxes[0]):int(boxes[2])])
                    output = predict(img_PIL)
                    predictions.append(output)
                else:
                    predictions.append(2)
            st.session_state.predictions = predictions
            st.session_state.trImage = trImage
            st.session_state.low_slice = low_sliceL
            st.session_state.low_slicer = low_sliceR
            st.session_state.total_slices_L = total_slices_L
            st.session_state.positive_slices = [i for i, pred in enumerate(predictions) if pred == 1]
            st.session_state.negtive_slices = [i for i, pred in enumerate(predictions) if pred == 0]
            st.session_state.current_pos_idx = 0 if st.session_state.positive_slices else -1
            st.session_state.current_neg_idx = 0 if st.session_state.negtive_slices else -1
            st.session_state.analysis_complete = True
            st.rerun()
        if st.session_state.analysis_complete:
            # 显示总体结果
            positive_number = st.session_state.predictions.count(1)
            negtive_number = st.session_state.predictions.count(0)
            total_result = 100 * positive_number/(positive_number + negtive_number)
            st.write(f"TMJ Disorders Probability: {total_result:.2f}%")
            st.write(f"Found {len(st.session_state.positive_slices)} slices with high risk of TMJ disorders")

            # 显示切片导航和图像
            if st.session_state.positive_slices:
                col1, col2, col3 = st.columns(3)
                
                # 上一个切片按钮
                if col1.button("Previous slice"):
                    st.session_state.current_pos_idx = (st.session_state.current_pos_idx - 1) % len(st.session_state.positive_slices)
                    st.rerun()
                
                # 下一个切片按钮
                if col2.button("Next slice"):
                    st.session_state.current_pos_idx = (st.session_state.current_pos_idx + 1) % len(st.session_state.positive_slices)
                    st.rerun()

                # 重置按钮
                if col3.button("Reset"):
                    for key in st.session_state.keys():
                        del st.session_state[key]
                    st.rerun()

                # 显示当前切片
                current_slice_idx = st.session_state.positive_slices[st.session_state.current_pos_idx]
                if current_slice_idx <= st.session_state.total_slices_L:
                    actual_slice_num = current_slice_idx + st.session_state.low_slice
                    st.write(f"Currently displaying slice number on the left side: {actual_slice_num} (The {st.session_state.current_pos_idx + 1}/{len(st.session_state.positive_slices)}-th  abnormal slice)")
                else:
                    actual_slice_num = current_slice_idx - st.session_state.total_slices_L + st.session_state.low_slicer
                    st.write(f"Currently displaying slice number on the right side: {actual_slice_num} (The {st.session_state.current_pos_idx + 1}/{len(st.session_state.positive_slices)}-th  abnormal slice)")
                
                # 显示图像
                img_out = cv2.flip(st.session_state.trImage[:, :, actual_slice_num], -1)
                st.image(Image.fromarray(np.uint8(img_out)).convert('RGB'), 
                        caption=f"slice {actual_slice_num}")
            else:
                st.write("No abnormal slices were found")
                # 显示切片导航和图像
                if st.session_state.negtive_slices:
                    col1, col2, col3 = st.columns(3)

                    # 上一个切片按钮
                    if col1.button("Previous slice"):
                        st.session_state.current_neg_idx = (st.session_state.current_neg_idx - 1) % len(
                            st.session_state.negtive_slices)
                        st.rerun()

                    # 下一个切片按钮
                    if col2.button("Next slice"):
                        st.session_state.current_neg_idx = (st.session_state.current_neg_idx + 1) % len(
                            st.session_state.negtive_slices)
                        st.rerun()

                    # 重置按钮
                    if col3.button("Reset"):
                        for key in st.session_state.keys():
                            del st.session_state[key]
                        st.rerun()

                    # 显示当前切片
                    current_slice_idx = st.session_state.negtive_slices[st.session_state.current_neg_idx]
                    if current_slice_idx <= st.session_state.total_slices_L:
                        actual_slice_num = current_slice_idx + st.session_state.low_slice
                        st.write(
                            f"Currently displaying slice number on the left side: {actual_slice_num} (The {st.session_state.current_pos_idx + 1}/{len(st.session_state.positive_slices)}-th  abnormal slice)")
                    else:
                        actual_slice_num = current_slice_idx - st.session_state.total_slices_L + st.session_state.low_slicer
                        st.write(f"Currently displaying slice number on the right side: {actual_slice_num} (The {st.session_state.current_pos_idx + 1}/{len(st.session_state.positive_slices)}-th  abnormal slice)")

                    # 显示图像
                    img_out = cv2.flip(st.session_state.trImage[:, :, actual_slice_num], -1)
                    st.image(Image.fromarray(np.uint8(img_out)).convert('RGB'),
                             caption=f"slice {actual_slice_num}")
