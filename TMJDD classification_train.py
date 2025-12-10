import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from timm import create_model
import os
from PIL import Image


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

        if self.transform:
            image = self.transform(image)

        return image, label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = MedicalDataset('./dataset/TMD_yolo_pre/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = MedicalDataset('./dataset/TMD_yolo_pre/test', transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# model = create_model('mobilenetv3_small_100', pretrained=False,
#                           pretrained_cfg={'file':r'./mode_pth/mobilenetv3_small_100_lamb-266a294c.pth'}, num_classes=2)

# model = create_model('efficientnet_b1', pretrained=False,
#                           pretrained_cfg={'file':r'./mode_pth/efficientnet_b1-533bc792.pth'}, num_classes=2)

# model = create_model('fastvit_t8', pretrained=False,
#                           pretrained_cfg={'file':'./mode_pth/fastvit_t8.pth.tar'}, num_classes=2)

# model = create_model('fastvit_t12', pretrained=False,
#                           pretrained_cfg={'file':'./mode_pth/fastvit_t12.pth.tar'}, num_classes=2)

# model = create_model('fastvit_s12', pretrained=False,
#                           pretrained_cfg={'file':'./mode_pth/fastvit_s12.pth.tar'}, num_classes=2)

# model = create_model('fastvit_sa36', pretrained=False,
#                           pretrained_cfg={'file':'./mode_pth/fastvit_sa36.pth.tar'}, num_classes=2)

# model = create_model('efficientvit_m0', pretrained=False,
#                           pretrained_cfg={'file':'./mode_pth/efficientvit_m0.pth'}, num_classes=2)

# model = create_model('efficientvit_m2', pretrained=False,
#                           pretrained_cfg={'file':'./mode_pth/efficientvit_m2.pth'}, num_classes=2)

model = create_model('efficientvit_m5', pretrained=False,
                          pretrained_cfg={'file':'./mode_pth/efficientvit_m5.pth'}, num_classes=2)


model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
num_epochs = 50

def train():
    model.train()
    best_test_accuracy = 0.0
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        accuracy = 100. * correct / total
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')

        if (epoch + 1) % 2 == 0:
            model.eval()
            test_loss = 0.0
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    test_total += labels.size(0)
                    test_correct += predicted.eq(labels).sum().item()
            test_accuracy = 100. * test_correct / test_total
            print(f'Epoch [{epoch + 1}/{num_epochs}], Test Accuracy: {test_accuracy:.2f}%')
            if test_accuracy > best_test_accuracy:
                best_test_accuracy = test_accuracy
                torch.save(model.state_dict(), './mode_path/TMD_evit_m5_med_classifier.pth')
        model.train()

if __name__ == '__main__':
    train()