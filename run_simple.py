import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# --- 1. DEFINE THE MODEL ARCHITECTURE ---
class SimpleBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "swin_base_patch4_window12_384", 
            pretrained=False, 
            features_only=True,
            out_indices=(0, 1, 2, 3),
            strict_img_size=False 
        )

    def forward(self, x):
        features = self.backbone(x)
        permuted_features = []
        # Expected channels for the 4 stages of Swin-Base
        expected_channels = [128, 256, 512, 1024]
        
        for i, f in enumerate(features):
            # If the last dimension matches the expected channels, it is NHWC.
            # We must flip it to NCHW.
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

class BiomedParseInference(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = SimpleBackbone()
        self.decoder = SimpleDecoder([128, 256, 512, 1024]) 

    def forward(self, x):
        features = self.backbone(x)
        logits = self.decoder(features)
        logits = F.interpolate(logits, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return torch.sigmoid(logits)

# --- 2. RUN INFERENCE ---
def main():
    print("🚀 Starting Simple Inference...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Device: {device}")

    model = BiomedParseInference().to(device)
    model.eval()

    WEIGHTS_PATH = "model_weights/biomedparse_v2.ckpt"
    if os.path.exists(WEIGHTS_PATH):
        print(f"⏳ Loading weights from {WEIGHTS_PATH}...")
        state_dict = torch.load(WEIGHTS_PATH, map_location=device)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        
        model.load_state_dict(state_dict, strict=False)
        print("✅ Weights loaded (partial match)")
    else:
        print(f"⚠️  Weights file not found at {WEIGHTS_PATH}. Using random weights.")

    INPUT_IMAGE = "test.jpg"
    if not os.path.exists(INPUT_IMAGE):
        print(f"❌ Error: {INPUT_IMAGE} not found.")
        return

    print(f"🔎 Processing {INPUT_IMAGE}...")
    img = Image.open(INPUT_IMAGE).convert("RGB")
    img = img.resize((384, 384)) 
    img_tensor = torch.tensor(np.array(img)).float().permute(2, 0, 1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        mask = output[0, 0].cpu().numpy()

    plt.imsave("simple_result.png", mask, cmap="jet")
    print("🎉 Done! Saved result to 'simple_result.png'")

if __name__ == "__main__":
    main()