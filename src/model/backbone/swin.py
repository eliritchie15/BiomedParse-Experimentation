import torch
import torch.nn as nn
import timm

class ShapeSpec:
    def __init__(self, channels=None, height=None, width=None, stride=None):
        self.channels = channels
        self.height = height
        self.width = width
        self.stride = stride

class D2SwinTransformer(nn.Module):
    def __init__(self, name, out_indices, drop_path_rate=0.2, pretrained=True):
        super().__init__()
        
        # Load the model from TIMM
        # We add strict_img_size=False to allow 1024x1024 inputs during inference
        self.backbone = timm.create_model(
            name, 
            pretrained=pretrained, 
            features_only=True, 
            out_indices=out_indices,
            drop_path_rate=drop_path_rate,
            strict_img_size=False 
        )
        
        # --- FIX: USE 384x384 DUMMY INPUT (Required by this model version) ---
        dummy_size = 384 
        dummy_input = torch.randn(1, 3, dummy_size, dummy_size)
        # ---------------------------------------------------------------------

        with torch.no_grad():
            features = self.backbone(dummy_input)
        
        self._out_feature_strides = {}
        self._out_feature_channels = {}
        self._out_features = ["res{}".format(i + 2) for i in range(len(out_indices))]
        
        for i, feat in enumerate(features):
            name = self._out_features[i]
            # Calculate stride dynamically based on the dummy input size
            stride = dummy_size // feat.shape[2]
            self._out_feature_strides[name] = stride
            self._out_feature_channels[name] = feat.shape[1]

    def forward(self, x):
        features = self.backbone(x)
        return {self._out_features[i]: f for i, f in enumerate(features)}

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], 
                stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }