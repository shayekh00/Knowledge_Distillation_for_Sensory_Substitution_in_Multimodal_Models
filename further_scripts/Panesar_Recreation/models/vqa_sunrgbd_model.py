import torch
import torch.nn as nn
import torchvision.models as models

class ImageEncoder(nn.Module):
    def __init__(self, use_pretrained=True):
        super(ImageEncoder, self).__init__()
        # Load VGG16 model
        # Using weights flag instead of pretrained boolean to avoid depreciation warnings if recent pytorch
        try:
            weights = models.VGG16_Weights.IMAGENET1K_V1 if use_pretrained else None
            vgg = models.vgg16(weights=weights)
        except AttributeError:
            # Fallback for older torchvision versions
            vgg = models.vgg16(pretrained=use_pretrained)
        
        # Remove the final classification layer (Linear 4096 -> 1000)
        # VGG.classifier is a Sequential:
        # (0): Linear(in_features=25088, out_features=4096, bias=True)
        # (1): ReLU(inplace=True)
        # (2): Dropout(p=0.5, ... )
        # (3): Linear(in_features=4096, out_features=4096, bias=True)
        # (4): ReLU(inplace=True)
        # (5): Dropout(p=0.5, ... )
        # (6): Linear(in_features=4096, out_features=1000, bias=True)
        
        # We keep up to the last hidden layer (index 5)
        self.features = vgg.features
        self.avgpool = vgg.avgpool
        self.classifier = nn.Sequential(*list(vgg.classifier.children())[:-1])
        
    def forward(self, x):
        """
        x: batch of RGB or Depth images (normalized)
        shape: (batch_size, 3, 224, 224)
        Returns size (batch_size, 4096)
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class QuestionEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size=300, hidden_size=512, out_size=4096):
        super(QuestionEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        
        # Two-layer LSTM
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=2, batch_first=True)
        
        # "The embeddings at each layer of this network is concatenated and 
        # passed through a fully-connected layer and tanh activation layer"
        self.fc = nn.Linear(2 * hidden_size, out_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        x shape: (batch_size, seq_length)
        Returns size (batch_size, out_size)
        """
        embedded = self.embedding(x)
        
        # LSTM returns output, (h_n, c_n)
        # h_n shape: (num_layers, batch_size, hidden_size)
        _, (h_n, _) = self.lstm(embedded)
        
        # Extract final hidden states from both layers and concatenate them
        # h_n[0] is layer 1 hidden state, h_n[1] is layer 2 hidden state
        layer1_h = h_n[0] # (batch_size, hidden_size)
        layer2_h = h_n[1] # (batch_size, hidden_size)
        concat_h = torch.cat((layer1_h, layer2_h), dim=1) # (batch_size, 2 * hidden_size)
        
        # Pass through FC and Tanh
        out = self.fc(concat_h) # (batch_size, out_size)
        out = self.tanh(out) # (batch_size, out_size)
        
        return out

class VQASUNRGBDModel(nn.Module):
    def __init__(self, vocab_size, num_classes=818, embed_size=300, hidden_size=512, fusion_method="conv1d"):
        """
        fusion_method: one of ['hadamard', 'addition', 'maxpool', 'conv1d', 'fusion_at_start']
        """
        super(VQASUNRGBDModel, self).__init__()
        self.fusion_method = fusion_method.lower()
        
        if self.fusion_method != "fusion_at_start":
            # Separate question encoders and separate image encoders
            self.q_enc_rgb = QuestionEncoder(vocab_size, embed_size, hidden_size, out_size=4096)
            self.q_enc_depth = QuestionEncoder(vocab_size, embed_size, hidden_size, out_size=4096)
            
            self.img_enc_rgb = ImageEncoder(use_pretrained=True)
            self.img_enc_depth = ImageEncoder(use_pretrained=True)
        else:
            # Single question encoder
            self.q_enc = QuestionEncoder(vocab_size, embed_size, hidden_size, out_size=4096)
            # Single combined image encoder
            self.img_enc = ImageEncoder(use_pretrained=True)
        
        if self.fusion_method == "conv1d":
            # 1D Convolution with 2 input channels (the stacked RGB and Depth tensors)
            # kernel size 1, 1 output channel.
            self.conv1d = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=1, stride=1, padding=0)
            
        # MLP network. 
        # The paper specifies an MLP ending with a softmax layer, but usually in PyTorch
        # softmax is subsumed by CrossEntropyLoss, so we limit to a Linear layer.
        self.mlp = nn.Sequential(
            nn.Linear(4096, num_classes)
        )

    def forward(self, img_rgb, img_depth, question):
        """
        img_rgb: (batch_size, 3, 224, 224) 
        img_depth: (batch_size, 3, 224, 224) - expected to be replicated to 3 channels to match VGG input
        question: (batch_size, seq_length)
        """
        if self.fusion_method == "fusion_at_start":
            # "combines the RGB and depth images ... via a pooling layer"
            stacked = torch.stack((img_rgb, img_depth), dim=0) # (2, B, 3, H, W)
            combined_img = torch.max(stacked, dim=0)[0]        # max pooling is a common standard pooling
            
            img_feat = self.img_enc(combined_img) # (B, 4096)
            q_feat = self.q_enc(question)         # (B, 4096)
            
            # Hadamard product
            h = q_feat * img_feat                 # (B, 4096)
            
        else:
            # Separate encodings
            f_rgb = self.img_enc_rgb(img_rgb)       # (B, 4096)
            f_depth = self.img_enc_depth(img_depth) # (B, 4096)
            
            g_rgb = self.q_enc_rgb(question)        # (B, 4096)
            g_depth = self.q_enc_depth(question)    # (B, 4096)
            
            # Question-Image Fusion (Hadamard product)
            h_rgb = g_rgb * f_rgb                   # (B, 4096)
            h_depth = g_depth * f_depth             # (B, 4096)
            
            # Fusing RGB and Depth Representations
            if self.fusion_method == "hadamard":
                h = h_rgb * h_depth
            elif self.fusion_method == "addition":
                h = h_rgb + h_depth
            elif self.fusion_method == "maxpool":
                h = torch.max(h_rgb, h_depth)
            elif self.fusion_method == "conv1d":
                # Stack h_rgb and h_depth to form (B, 2, 4096)
                h_stacked = torch.stack((h_rgb, h_depth), dim=1)
                # Pass through Conv1D
                h_conv = self.conv1d(h_stacked) # (B, 1, 4096)
                h = h_conv.squeeze(1)           # (B, 4096)
            else:
                raise ValueError(f"Unknown fusion method {self.fusion_method}")
        
        # MLP Network Prediction
        out = self.mlp(h) # (B, num_classes)
        return out

# To get predictions during inference you would take argmax:
# logits = model(img_rgb, img_depth, question)
# predicted_classes = torch.argmax(logits, dim=1)
