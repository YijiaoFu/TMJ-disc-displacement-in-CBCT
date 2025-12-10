import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from timm import create_model
import os
from PIL import Image
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, roc_curve, auc
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import pandas as pd
from collections import defaultdict
import numpy as np


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class MedicalDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.images = []
        self.labels = []
        pos_path = os.path.join(img_dir, 'class1')
        neg_path = os.path.join(img_dir, 'class0')

        for img_name in os.listdir(pos_path):
            self.images.append(os.path.join(pos_path, img_name))
            self.labels.append(1)
        for img_name in os.listdir(neg_path):
            self.images.append(os.path.join(neg_path, img_name))
            self.labels.append(0)
            
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        img_name = os.path.basename(img_path)
        patient_id = img_name.split('_')[0]
        
        if self.transform:
            image = self.transform(image)
        return image, label, patient_id

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
test_dataset_path = './dataset/TMD_yolo_pre/test'
test_dataset = MedicalDataset(test_dataset_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

model_evitm5 = create_model('efficientvit_m5', pretrained=False, num_classes=2)
checkpoint = torch.load('./mode_path/TMD_evit_m5_med_classifier.pth',
                        map_location=torch.device('cpu'))
model_evitm5.load_state_dict(checkpoint)
model_evitm5 = model_evitm5.to(device)


def test_and_collect_by_patient(model, data_loader):
    model.eval()
    patient_results = defaultdict(lambda: {'probs': [], 'preds': [], 'label': -1})
    with torch.no_grad():
        for images, labels, patient_ids in data_loader:
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            for i in range(len(patient_ids)):
                p_id = patient_ids[i]
                patient_results[p_id]['probs'].append(probabilities[i])
                patient_results[p_id]['preds'].append(predictions[i])
                if patient_results[p_id]['label'] == -1:
                    patient_results[p_id]['label'] = labels[i].item()    
    return patient_results
slice_results_by_patient = test_and_collect_by_patient(model_evitm5, test_loader)

y_true_patient = []
y_pred_patient = []
y_prob_patient = []

for patient_id, results in slice_results_by_patient.items():
    true_label = results['label']
    slice_preds = np.array(results['preds'])
    patient_prob = np.mean(slice_preds) 
    patient_pred = 1 if patient_prob >= 0.5 else 0
    y_true_patient.append(true_label)
    y_pred_patient.append(patient_pred)
    y_prob_patient.append(patient_prob)

results_df = pd.DataFrame(columns=['Model', 'Acc', 'Recall', 'Precision', 'Sensitivity', 'Specificity', 'AUC', 'AUPR','TP','TN','FP','FN'])

y_true = np.array(y_true_patient)
y_pred = np.array(y_pred_patient)
y_prob = np.array(y_prob_patient)

acc = accuracy_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
sensitivity = recall
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
auc_score = roc_auc_score(y_true, y_prob)
aupr_score = average_precision_score(y_true, y_prob)

model_name = 'efficientvit_m5_patient_level'
results_df = pd.concat([
    results_df,
    pd.DataFrame({
        'Model': [model_name],
        'Acc': [acc],
        'Recall': [recall],
        'Precision': [precision],
        'Sensitivity': [sensitivity],
        'Specificity': [specificity],
        'AUC': [auc_score],
        "AUPR": [aupr_score],
        "TP": [tp],      
        "TN": [tn],  
        "FP": [fp],  
        "FN": [fn],                   
    })
], ignore_index=True)

# 保存结果
output_file = './results/TMD_patient_level_evaluation_results.csv'
results_df.to_csv(output_file, index=False)