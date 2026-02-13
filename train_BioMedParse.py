import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image, ImageDraw
import os
import glob

# --- 1. DEFINE THE MODEL ARCHITECTURE ---
class SimpleBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "swin_base_patch4_window12_384", 
            pretrained=True, 
            features_only=True,
            out_indices=(0, 1, 2, 3),
            strict_img_size=False 
        )

    def forward(self, x):
        features = self.backbone(x)
        permuted_features = []
        expected_channels = [128, 256, 512, 1024]
        for i, f in enumerate(features):
            if f.shape[-1] == expected_channels[i]:
                permuted_features.append(f.permute(0, 3, 1, 2))
            else:
                permuted_features.append(f)
        return permuted_features

class SimpleDecoder(nn.Module):
    def __init__(self, in_channels_list):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Conv2d(in_c, 256, kernel_size=1) for in_c in in_channels_list
        ])
        self.fusion = nn.Conv2d(256 * 4, 256, kernel_size=1)
        self.predictor = nn.Conv2d(256, 1, kernel_size=1)

    def forward(self, features):
        target_size = features[0].shape[-2:]
        resized_feats = []
        for i, feat in enumerate(features):
            proj = self.projections[i](feat)
            resized = F.interpolate(proj, size=target_size, mode='bilinear', align_corners=False)
            resized_feats.append(resized)
        concat = torch.cat(resized_feats, dim=1)
        fused = self.fusion(concat)
        logits = self.predictor(fused)
        return logits

class BiomedParseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = SimpleBackbone()
        self.decoder = SimpleDecoder([128, 256, 512, 1024]) 

    def forward(self, x):
        features = self.backbone(x)
        logits = self.decoder(features)
        logits = F.interpolate(logits, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return logits 

# --- 2. THE YOLO DATASET LOADER (UPDATED FOR SUBFOLDERS) ---
class YoloDataset(Dataset):
    def __init__(self, root_dir, split="train", img_size=384):
        # Automatically finds dataset/train/images or dataset/valid/images
        self.img_dir = os.path.join(root_dir, split, "images")
        self.label_dir = os.path.join(root_dir, split, "labels")
        self.img_size = img_size
        
        # Support both jpg and png
        self.images = sorted(glob.glob(os.path.join(self.img_dir, "*.jpg")) + 
                             glob.glob(os.path.join(self.img_dir, "*.png")))
        
        if len(self.images) == 0:
            print(f"⚠️  No images found in {self.img_dir}. Check your folders!")
        else:
            print(f"✅ Found {len(self.images)} images in '{split}' set.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        img_resized = img.resize((self.img_size, self.img_size))
        
        # Construct label path
        # Checks for .txt with same name as image
        basename = os.path.basename(img_path)
        txt_name = os.path.splitext(basename)[0] + ".txt"
        label_path = os.path.join(self.label_dir, txt_name)
        
        # Create Black Mask
        mask = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(mask)
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = list(map(float, line.strip().split()))
                    # YOLO: class x_center y_center width height (normalized 0-1)
                    if len(parts) >= 5:
                        _, x_c, y_c, bw, bh = parts
                        
                        x1 = (x_c - bw/2) * w
                        y1 = (y_c - bh/2) * h
                        x2 = (x_c + bw/2) * w
                        y2 = (y_c + bh/2) * h
                        
                        draw.rectangle([x1, y1, x2, y2], fill=255)
        
        mask_resized = mask.resize((self.img_size, self.img_size), resample=Image.NEAREST)

        img_t = torch.tensor(np.array(img_resized)).float().permute(2, 0, 1) / 255.0
        mask_t = torch.tensor(np.array(mask_resized)).float().unsqueeze(0) / 255.0
        mask_t = (mask_t > 0.5).float() # Ensure binary 0/1
        
        return img_t, mask_t

# --- 3. TRAINING LOOP ---
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on {device}...")

    # CONFIG
    DATASET_ROOT = "dataset" # The parent folder containing train/valid/test
    BATCH_SIZE = 4
    EPOCHS = 10
    LR = 1e-4

    # Setup Model
    model = BiomedParseModel().to(device)
    
    # Load Backbone if available
    WEIGHTS_PATH = "model_weights/biomedparse_v2.ckpt"
    if os.path.exists(WEIGHTS_PATH):
        print("⏳ Loading Backbone Weights...")
        state_dict = torch.load(WEIGHTS_PATH, map_location=device)
        if "state_dict" in state_dict: state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict=False)

    # Prepare Data Loaders
    train_ds = YoloDataset(DATASET_ROOT, split="train")
    valid_ds = YoloDataset(DATASET_ROOT, split="valid")
    
    if len(train_ds) == 0:
        print("❌ Cannot train without data. Exiting.")
        return

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    # Validation doesn't need shuffle
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False) if len(valid_ds) > 0 else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    print(f"\n🥊 Starting {EPOCHS} Epochs!")
    
    for epoch in range(EPOCHS):
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0
        for i, (imgs, masks) in enumerate(train_loader):
            imgs, masks = imgs.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            if i % 10 == 0:
                print(f"   [Epoch {epoch+1}] Batch {i} | Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / len(train_loader)
        print(f"✅ Epoch {epoch+1} Training Loss: {avg_train_loss:.4f}")

        # --- VALIDATION PHASE (Optional) ---
        if valid_loader:
            model.eval()
            valid_loss = 0
            with torch.no_grad():
                for imgs, masks in valid_loader:
                    imgs, masks = imgs.to(device), masks.to(device)
                    outputs = model(imgs)
                    loss = criterion(outputs, masks)
                    valid_loss += loss.item()
            print(f"   🔍 Validation Loss: {valid_loss / len(valid_loader):.4f}")

    # Save
    torch.save(model.state_dict(), "yolo_trained_model.pth")
    print("\n🎉 Training Complete! Saved 'yolo_trained_model.pth'")

if __name__ == "__main__":
    train()