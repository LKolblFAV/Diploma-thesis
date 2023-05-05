from PIL import Image
from tqdm import tqdm
import gc
import torch
torch.cuda.is_available()
from GPUtil import showUtilization as gpu_usage
import wandb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedGroupKFold
import time
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from numba import cuda
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import albumentations as A
# import cv2
from albumentations.pytorch import ToTensorV2
import numpy as np
import copy
import random
import timm

hyperparameter_defaults = dict(
    alpha = 0.25,
    gamma = 2,
    learning_rate = 0.01,
    num_epochs = 1,
    optimizer = "adam",
    )
#wandb.init(config=hyperparameter_defaults)
#config = wandb.config


wandb.login()
wandb.init(project="")

train_features = pd.read_csv("/auto/plzen1/home/scrower/data_competition/train_features.csv", index_col="id")
test_features = pd.read_csv("/auto/plzen1/home/scrower/data_competition/test_features.csv", index_col="id")
train_labels = pd.read_csv("/auto/plzen1/home/scrower/data_competition/train_labels.csv", index_col="id")

species_labels = sorted(train_labels.columns.unique())


sgkf = StratifiedGroupKFold(n_splits=5)
list_of_sites = []
for row in train_features.itertuples(index=False):
    for k,v in enumerate(row):
        konec = v[-3:]
        if konec == "jpg":
            pass
        else:
            list_of_sites.append(konec)
# print(list_of_sites)
list_of_ids = []
for row in train_features.itertuples(index=False):
    for k,v in enumerate(row):
        image_ids = v[15:23]
        if image_ids == "":
            pass
        else:
            list_of_ids.append(image_ids)
# print(list_of_ids)
list_of_class_numbers = []

counterr = 0
for row in train_labels.itertuples(index=False):
    for a,b in enumerate(row):
        if b == 1.0:
            list_of_class_numbers.append(counterr)
        counterr = counterr + 1
    counterr = 0

x = list_of_ids
y = list_of_class_numbers
groups = list_of_sites
train_fold = [[],[],[],[],[]]
test_fold = [[],[],[],[],[]]
counter = 0;
for train, test in sgkf.split(x, y, groups=groups):
    train_fold[counter] = train
    test_fold[counter] = test
    counter = counter+1

x_train1 =[]
y_train1 =[]
x_eval1 = []
y_eval1 = []
for j in range(5):
    d = train_features
    d1 = train_labels
    y0 = train_features
    y1 = train_labels

    d = d.drop(['site'], axis=1)
    y0 = y0.drop(['site'], axis=1)
    index = d.index

    list_of_ids_infold = []
    for i in train_fold[j]:
        list_of_ids_infold.append(list_of_ids[i])

    rozdil = []
    rozdil = sorted(list(set(index) - set(list_of_ids_infold)))

    for i in rozdil:    
        d = d.drop(i)
        d1 = d1.drop(i)
    for i in list_of_ids_infold:
        y0 = y0.drop(i)
        y1 = y1.drop(i)
    x_train1.append(d)
    y_train1.append(d1)
    x_eval1.append(y0)
    y_eval1.append(y1)

class ImagesDataset(Dataset):
    """Reads in an image, transforms pixel values, and serves
    a dictionary containing the image id, image tensors, and label.
    """

    def __init__(self, x_df, y_df=None,device='cuda',transform=None):
        self.data = x_df
        self.label = y_df
        self.transform = transform
        

    def __getitem__(self, index,device='cuda'):
        image = Image.open("/auto/plzen1/home/scrower/data_competition/" + self.data.iloc[index]["filepath"]).convert("RGB")
        # Convert PIL image to numpy array
        image_np = np.array(image)
        # Apply transformations
        image = self.transform(image=image_np)["image"]
        image_id = self.data.index[index]
        # if we don't have labels (e.g. for test set) just return the image and image id
        if self.label is None:
            sample = {"image_id": image_id, "image": image}
        else:
            label = torch.tensor(self.label.iloc[index].values, 
                                 dtype=torch.float)
            sample = {"image_id": image_id, "image": image, "label": label}
        return sample

    def __len__(self):
        return len(self.data)

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.GaussianBlur(blur_limit = (3,7) ,p=0.2),
    A.OneOf([
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=1),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2,p=1),
        ], p=0.5),
    A.OneOf([
            A.HueSaturationValue(p=0.6,hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=50),
            A.RGBShift(p=0.4),
        ], p=0.2),
    A.Equalize (mode='pil', by_channels=True, mask=None, mask_params=(), always_apply=False, p=0.05),
    A.Resize(256, 256),
    A.CenterCrop(224,224), 
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(224, 224), 
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
])

def Cutout(num_images,length,batch,holes):
    #num_images = 2 # number of images
    #length = 40 # length(int) in pixels of each square
    batch_size = batch["label"].size(0)
    if batch_size < num_images:
        print("end")
        return(batch)
    num_holes = holes
    re_mask = []
    masks = []
    ran = random.sample(range(batch_size), num_images)
    h = 224 #height of image
    w = 224 #width of image
    mask = np.ones((h, w), np.float32)
    for n in range(num_images):
        for i in range(num_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - length // 2, 0, h)
            y2 = np.clip(y + length // 2, 0, h)
            x1 = np.clip(x - length // 2, 0, w)
            x2 = np.clip(x + length // 2, 0, w)
            mask[y1: y2, x1: x2] = 0.
        masks.append(mask)
        mask = np.ones((h, w), np.float32)
        
        
    for i in range(num_images):
        mask = torch.from_numpy(masks[i])
        mask = mask.expand_as(batch["image"][ran[i]])
        batch["image"][ran[i]] = batch["image"][ran[i]] * mask
    return(batch)

def Mixup(num_pairs,batch): # returns same size batch and float labels
    #num_pairs = 2 # number of pairs to change
    batch_size = batch["label"].size(0)
    if batch_size < num_pairs*2:
        print("end")
        return(batch)
    label_ratio = []
    ran = random.sample(range(batch_size), num_pairs*2)
    h = 224 #height of image
    w = 224 #width of image
    for i in range(num_pairs):
        img = batch["image"][ran[2*i]]
        img2 = batch["image"][ran[(2*i)+1]]
        r = np.random.beta(4.0, 4.0)  # mixup ratio, alpha=beta=32.0
        im = img * r + img2 * (1 - r)
        label_0 = batch["label"][ran[(2*i)]] * r + batch["label"][ran[(2*i)+1]] * (1 - r)
        r = np.random.beta(4.0, 4.0)  # mixup ratio, alpha=beta=32.0
        im2 = img2 * r + img * (1 - r)
        label_1 = batch["label"][ran[(2*i)+1]] * r + batch["label"][ran[(2*i)]] * (1 - r)
        batch["image"][ran[(2*i)]] = im
        img2 = batch["image"][ran[(2*i)+1]] = im2
        batch["label"][ran[(2*i)]] = label_0
        batch["label"][ran[(2*i)+1]] = label_1
    return(batch)

def Mixcut(num_pairs,length,batch):
    #num_pairs = 2 # Number of pairs from batch
    #length = 100 # length(int) in pixels
    batch_size = batch["label"].size(0)
    if batch_size < num_pairs*2:
        print("end")
        return(batch)
    label_ratio = []
    ran = random.sample(range(batch_size), num_pairs*2)
    
    changed_labels = []
    h = 224 #height of image
    w = 224 #width of image
    img_area = h*w
    mask = np.ones((h, w), np.float32)
    for n in range(num_pairs):
        y = np.random.randint(h)
        x = np.random.randint(w)
        
        y1 = np.clip(y - length // 2, 0, h)
        y2 = np.clip(y + length // 2, 0, h)
        x1 = np.clip(x - length // 2, 0, w)
        x2 = np.clip(x + length // 2, 0, w)
        
        y0 = random.randrange(-y1,h-y2)
        x0 = random.randrange(-x1,w-x2)
        y3 = random.randrange(-y1,h-y2)
        x3 = random.randrange(-x1,w-x2)
        mask_height = y2-y1
        mask_width = x2-x1
        area = mask_height * mask_width
        img_left = img_area - area
        ratio = img_left / img_area
        
        img = torch.clone(batch["image"][ran[(2*n)]])
        imga = torch.clone(batch["image"][ran[(2*n)+1]])
        batch["image"][ran[(2*n)]][:,y1: y2,x1: x2] = imga[:,y1+y0: y2+y0,x1+x0: x2+x0]
        batch["image"][ran[(2*n)+1]][:,y1: y2,x1: x2] = img[:,y1+y3: y2+y3,x1+x3: x2+x3]
        label_0 = batch["label"][ran[(2*n)]] * ratio + batch["label"][ran[(2*n)+1]] * (1 - ratio)
        label_1 = batch["label"][ran[(2*n)+1]] * ratio + batch["label"][ran[(2*n)]] * (1 - ratio)
        batch["label"][ran[(2*n)]] = label_0
        batch["label"][ran[(2*n)+1]] = label_1
    return(batch)

train_dataset = ImagesDataset(x_train1[0], y_train1[0],transform=train_transform)
train_dataloader = DataLoader(train_dataset, batch_size=32,shuffle=True)

from torch import nn
import torchvision.models as models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = timm.create_model('resnext50_32x4d', pretrained=True, num_classes=8)
num_in_features = model.get_classifier().in_features;
model.fc = nn.Sequential(
    nn.Linear(
        num_in_features, 8
    ),  # final dense layer outputs 8-dim corresponding to our target classes
)
net = model
net = net.cuda()
next(net.parameters()).is_cuda #returns a bool value, True - your model is truly on GPU, False - it is not

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.007732227549238594, momentum=0.9)


eval_dataset = ImagesDataset(x_eval1[0], y_eval1[0],transform=val_transform)
eval_dataloader = DataLoader(eval_dataset, batch_size=8)
eval_true = y_eval1[0].idxmax(axis=1)
from sklearn.metrics import log_loss

#if config.optimizer=='sgd':
#    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9)
#elif config.optimizer=='adam':
#    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
#elif config.optimizer=='adamw':
#    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)


print_frequency = 100;
running_loss = 0.0
num_epochs = 50
totalTrainAccuracy = 0
tracking_loss = {}
min_val_loss = np.inf
min_com_loss = np.inf
max_acc = np.NINF
alpha = 0.6
gamma = 5

for epoch in range(1, num_epochs + 1):
    print(f"Starting epoch {epoch}")

    totalTrainAccuracy = 0
    trainAccuracy = 0
    # iterate through the dataloader batches. tqdm keeps track of progress.
    for batch_n, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):

        # 1) zero out the parameter gradients so that gradients from previous batches are not used in this step
        optimizer.zero_grad()
        #batch = Mixup(2,batch)
        #batch = Mixcut(2,70,batch)
        #batch = Cutout(4,70,batch,1)
        # 2) run the foward step on this batch of images
        outputs = model(batch["image"].to(device))  # to.(device)
        # 3) compute the loss
        cel = criterion(outputs, batch["label"].to(device))
        pt = torch.exp(-cel)
        focal_loss = (alpha * (1-pt)**gamma * cel) # mean over the batch
        loss = focal_loss        # let's keep track of the loss by epoch and batch
        # tracking_loss[(epoch, batch_n)] = float(loss)

        # 4) compute our gradients
        loss.backward()
        # update our weights
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        trainAccuracy = (predicted == torch.max(batch["label"].to(device), 1)[1]).float().sum().item()
        trainAccuracy = 100 * trainAccuracy / batch["label"].size(0)
        totalTrainAccuracy += trainAccuracy

        # print(f'predicted: {predicted},image label: {batch["label"]} novy tensor:{torch.max(batch["label"].to(device),1)[1]}')
        # print(f'[{epoch + 1}, {batch_n + 1:5d}] labelsize: {batch["label"].size(0)} trainacc: {trainAccuracy}')
        # print(f'[{epoch + 1}, {batch_n + 1:5d}] trainAccu: {trainAccuracy}')

        running_loss += loss.item()
        if batch_n % print_frequency == (print_frequency - 1):
            wandb.log({'Train_loss': (running_loss / print_frequency)})
            wandb.log({'Epoch': epoch})
            wandb.log({'Train_accuracy': totalTrainAccuracy / print_frequency})
           # print(f'[{epoch}, {batch_n + 1:5d}] loss: {running_loss / print_frequency:.3f}')
           # print(f'[{epoch}, {batch_n + 1:5d}] AvgtrainAccu: {totalTrainAccuracy / print_frequency:.3f}')
            running_loss = 0.0
            totalTrainAccuracy = 0
    # gc.collect()
    # torch.cuda.empty_cache()

    preds_collector = []
    valid_loss = 0.0
    # put the model in eval mode so we don't update any parameters
    model.eval()
    # we aren't updating our weights so no need to calculate gradients
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
            # 1) run the forward step
            logits = model.forward(batch["image"].to(device))
            # 2) apply softmax so that model outputs are in range [0,1]
            preds = nn.functional.softmax(logits, dim=1)
            # 3) store this batch's predictions in df
            val_loss = criterion(logits, batch["label"].to(device))
            pt = torch.exp(-val_loss)
            focal_loss = (alpha * (1-pt)**gamma * val_loss) # mean over the batch
            val_loss = focal_loss
            valid_loss += val_loss.item()
            # note that PyTorch Tensors need to first be detached from their computational graph before converting to numpy arrays
            preds_df = pd.DataFrame(
                preds.cpu().detach().numpy(),
                index=batch["image_id"],
                columns=species_labels,
            )
            preds_collector.append(preds_df)

    eval_preds_df = pd.concat(preds_collector)

    eval_predictions = eval_preds_df.idxmax(axis=1)
    correct = (eval_predictions == eval_true).sum()
    accuracy = correct / len(eval_predictions)
    validation_loss = valid_loss / len(eval_dataloader)
    competition_loss = log_loss(eval_true,eval_preds_df)
    model.train()
    wandb.log({'Validation_Accuracy': accuracy})
    wandb.log({'Validation_Loss': validation_loss})
    wandb.log({'Competition_Loss': competition_loss})
    if min_val_loss > validation_loss:
        min_val_loss = validation_loss
        best_model_vloss = copy.deepcopy(model)
    if min_com_loss > competition_loss:
        min_com_loss = competition_loss
        best_model_closs = copy.deepcopy(model)
    if max_acc < accuracy:
        max_acc = accuracy
        best_model_acc = copy.deepcopy(model)


torch.save(best_model_acc, "/auto/plzen1/home/scrower/Model_acc.pth")
#torch.save(best_model_vloss, "/auto/plzen1/home/scrower/sweeped_para_8_vloss.pth")
torch.save(best_model_closs, "/auto/plzen1/home/scrower/Model_closs.pth")