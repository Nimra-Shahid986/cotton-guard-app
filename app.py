"""
Cotton Leaf Disease Detection System
=====================================
A deep learning application for Pakistani cotton farmers to detect
diseases in cotton leaves using Swin Transformer & ConvNeXt models.

Datasets:
  - SAR-CLD 2024 (7 classes) → Swin-T
  - Cotton Leaf Disease (4 classes) → ConvNeXt-T
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import time
import io

# ─── Page Config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cotton Guard — Leaf Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global */
    .stApp {
        background: linear-gradient(135deg, #0a0f1a 0%, #0d1525 40%, #0f1a2e 100%);
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Header */
    .app-header {
        text-align: center;
        padding: 2rem 0 1rem;
    }
    .app-header h1 {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 800;
        font-size: 2.4rem;
        background: linear-gradient(135deg, #4ade80 0%, #22d3ee 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .app-header p {
        color: #94a3b8;
        font-size: 1rem;
        margin-top: 0.3rem;
        font-weight: 400;
    }

    /* Card panels */
    .glass-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(100, 200, 180, 0.12);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(20px);
    }
    .glass-card-header {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        font-weight: 500;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 1rem;
    }

    /* Prediction result */
    .prediction-box {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(6, 182, 212, 0.08));
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        margin: 0.8rem 0;
    }
    .prediction-label {
        font-size: 0.7rem;
        font-weight: 600;
        color: #10b981;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .prediction-name {
        font-size: 1.8rem;
        font-weight: 800;
        color: #f0fdf4;
        margin: 0.2rem 0;
    }
    .prediction-index {
        font-size: 0.85rem;
        color: #94a3b8;
    }

    /* Disease result */
    .disease-box {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(251, 146, 60, 0.08));
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        margin: 0.8rem 0;
    }
    .disease-label {
        font-size: 0.7rem;
        font-weight: 600;
        color: #ef4444;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    /* Confidence ring */
    .confidence-section {
        text-align: center;
        padding: 1rem 0;
    }
    .confidence-pct {
        font-size: 2.8rem;
        font-weight: 800;
        color: #4ade80;
        line-height: 1;
    }

    /* Metric cards */
    .metric-row {
        display: flex;
        gap: 0.8rem;
        flex-wrap: wrap;
        margin: 1rem 0;
    }
    .metric-card {
        flex: 1;
        min-width: 120px;
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(100, 200, 180, 0.08);
        border-radius: 10px;
        padding: 0.8rem;
        text-align: center;
    }
    .metric-card .metric-label {
        font-size: 0.65rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 500;
    }
    .metric-card .metric-value {
        font-size: 1rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-top: 0.2rem;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Probability bars */
    .prob-item {
        display: flex;
        align-items: center;
        margin: 0.5rem 0;
        gap: 0.8rem;
    }
    .prob-name {
        width: 180px;
        font-size: 0.82rem;
        color: #cbd5e1;
        font-weight: 500;
        text-align: right;
        flex-shrink: 0;
    }
    .prob-bar-bg {
        flex: 1;
        height: 8px;
        background: rgba(30, 41, 59, 0.8);
        border-radius: 4px;
        overflow: hidden;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.5s ease;
    }
    .prob-pct {
        width: 50px;
        font-size: 0.78rem;
        color: #94a3b8;
        font-family: 'JetBrains Mono', monospace;
        text-align: right;
        flex-shrink: 0;
    }

    /* Chatbot styles */
    .chat-container {
        position: fixed;
        bottom: 90px;
        right: 20px;
        width: 380px;
        max-height: 520px;
        background: #0d1525;
        border: 1px solid rgba(100, 200, 180, 0.2);
        border-radius: 16px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.5);
        z-index: 9999;
        display: flex;
        flex-direction: column;
        overflow: hidden;
    }
    .chat-header {
        background: linear-gradient(135deg, #10b981, #06b6d4);
        padding: 1rem 1.2rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .chat-header-title {
        font-weight: 700;
        color: white;
        font-size: 0.95rem;
    }
    .chat-header-sub {
        color: rgba(255,255,255,0.8);
        font-size: 0.72rem;
    }

    /* Upload area */
    .upload-zone {
        border: 2px dashed rgba(100, 200, 180, 0.25);
        border-radius: 14px;
        padding: 2rem;
        text-align: center;
        background: rgba(15, 23, 42, 0.4);
        transition: border-color 0.3s;
    }
    .upload-zone:hover {
        border-color: rgba(100, 200, 180, 0.5);
    }

    /* Disease info section */
    .info-card {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(100, 200, 180, 0.08);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
    }
    .info-card h4 {
        color: #4ade80;
        font-size: 0.9rem;
        margin: 0 0 0.4rem 0;
    }
    .info-card p {
        color: #94a3b8;
        font-size: 0.82rem;
        margin: 0;
        line-height: 1.5;
    }

    /* Hide Streamlit defaults */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    header[data-testid="stHeader"] {background: transparent;}

    /* Streamlit overrides */
    .stSelectbox label, .stFileUploader label {
        color: #94a3b8 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.7rem !important;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    div[data-testid="stFileUploader"] {
        background: rgba(15, 23, 42, 0.4);
        border: 2px dashed rgba(100, 200, 180, 0.2);
        border-radius: 14px;
        padding: 1rem;
    }
    div[data-testid="stFileUploader"]:hover {
        border-color: rgba(100, 200, 180, 0.4);
    }
</style>
""", unsafe_allow_html=True)


# ─── Constants ─────────────────────────────────────────────────────────────
SAR_CLD_CLASSES = [
    "Bacterial Blight",
    "Curl Virus",
    "Healthy Leaf",
    "Herbicide Growth Damage",
    "Leaf Hopper Jassids",
    "Leaf Redding",
    "Leaf Variegation",
]

COTTON_LEAF_CLASSES = [
    "Bacterial Blight",
    "Curl Virus",
    "Fussarium Wilt",
    "Healthy",
]

DATASET_INFO = {
    "SAR-CLD 2024 — 7 Classes": {
        "classes": SAR_CLD_CLASSES,
        "model_file": "models/swin_t_best.pt",
        "architecture": "LDASN (Lightweight Dynamic Attention)",
        "arch_key": "Swin_T",
        "img_size": 64,
        "description": "SAR-CLD 2024 dataset with 7 cotton leaf disease classes. Best performer: Swin Transformer.",
    },
    "Cotton Leaf Disease — 4 Classes": {
        "classes": COTTON_LEAF_CLASSES,
        "model_file": "models/convnext_t_best.pt",
        "architecture": "ConvNeXt Tiny (ConvNeXt-T)",
        "arch_key": "ConvNeXt_T",
        "img_size": 224,
        "description": "Cotton Leaf Disease dataset with 4 classes. Best performer: ConvNeXt-T.",
    },
}

# Disease information database
DISEASE_INFO = {
    "Bacterial Blight": {
        "severity": "High",
        "description": "Angular water-soaked lesions on leaves that turn brown. Causes defoliation and boll rot.",
        "symptoms": "Water-soaked angular spots, blackening of veins, premature defoliation.",
        "treatment": "Use copper-based bactericides. Plant resistant varieties. Remove and destroy infected debris. Ensure proper field drainage.",
        "prevention": "Use certified disease-free seeds, crop rotation with non-host crops, avoid overhead irrigation.",
    },
    "Curl Virus": {
        "severity": "Very High",
        "description": "Transmitted by whiteflies, causes upward or downward curling of leaves, stunted growth, and severe yield loss.",
        "symptoms": "Leaf curling, thickened veins, enation (leaf-like outgrowths), stunted plants, reduced boll formation.",
        "treatment": "Control whitefly vectors with appropriate insecticides (imidacloprid, acetamiprid). Remove infected plants early. Use sticky traps.",
        "prevention": "Plant resistant varieties (e.g., BT cotton with CLCuV tolerance), early sowing, maintain field hygiene, border crops to trap whiteflies.",
    },
    "Healthy Leaf": {
        "severity": "None",
        "description": "The leaf appears healthy with no visible signs of disease or pest damage.",
        "symptoms": "No symptoms — uniform green color, normal leaf shape and size.",
        "treatment": "No treatment needed. Continue regular crop management practices.",
        "prevention": "Maintain balanced nutrition, proper irrigation scheduling, and regular scouting to catch issues early.",
    },
    "Healthy": {
        "severity": "None",
        "description": "The leaf appears healthy with no visible signs of disease or pest damage.",
        "symptoms": "No symptoms — uniform green color, normal leaf shape and size.",
        "treatment": "No treatment needed. Continue regular crop management practices.",
        "prevention": "Maintain balanced nutrition, proper irrigation scheduling, and regular scouting to catch issues early.",
    },
    "Herbicide Growth Damage": {
        "severity": "Medium",
        "description": "Damage caused by herbicide drift or misapplication, resulting in abnormal leaf growth patterns.",
        "symptoms": "Cupped or strapped leaves, abnormal growth, epinasty, chlorosis.",
        "treatment": "Foliar application of growth regulators. Provide adequate irrigation and nutrition to aid recovery.",
        "prevention": "Use proper herbicide application techniques, avoid spraying on windy days, maintain buffer zones, calibrate sprayers regularly.",
    },
    "Leaf Hopper Jassids": {
        "severity": "Medium-High",
        "description": "Jassids suck cell sap from leaves causing yellowing and curling of leaf margins.",
        "symptoms": "Yellowing of leaf margins, downward curling, 'hopper burn' in severe cases, leaf drying.",
        "treatment": "Apply systemic insecticides (thiamethoxam, imidacloprid). Use neem-based sprays for organic approach.",
        "prevention": "Use resistant varieties, intercropping, maintain natural predators, avoid excessive nitrogen fertilization.",
    },
    "Leaf Redding": {
        "severity": "Medium",
        "description": "Reddening of leaves due to nutrient deficiency (often magnesium) or physiological stress.",
        "symptoms": "Reddish-purple discoloration of leaves, starting from lower leaves and moving upward.",
        "treatment": "Foliar application of magnesium sulphate. Correct nutrient imbalances through soil testing and balanced fertilization.",
        "prevention": "Regular soil testing, balanced NPK application, ensure adequate potassium and magnesium levels.",
    },
    "Leaf Variegation": {
        "severity": "Medium",
        "description": "Irregular patches of different colors on leaves, often caused by viral infections or genetic factors.",
        "symptoms": "Mosaic patterns, irregular light and dark green patches, sometimes with yellow streaks.",
        "treatment": "Remove severely affected plants. Control insect vectors. No direct cure for viral variegation.",
        "prevention": "Use virus-free planting material, control aphid and whitefly vectors, maintain field hygiene.",
    },
    "Fussarium Wilt": {
        "severity": "High",
        "description": "A soil-borne fungal disease that blocks water-conducting vessels, causing wilting and death.",
        "symptoms": "Yellowing of leaves on one side, wilting despite adequate moisture, brown discoloration of stem vascular tissue.",
        "treatment": "No effective chemical cure once infected. Remove and destroy infected plants. Soil solarization. Apply Trichoderma-based biocontrol agents.",
        "prevention": "Use resistant varieties, long crop rotation (3+ years), avoid waterlogged conditions, soil solarization before planting.",
    },
}


# ─── Custom LDASN Architecture (SAR-CLD 2024 Model) ───────────────────────
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size, stride, padding, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return self.bn(self.pw(self.dw(x)))


class MultiScaleExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
        )
        self.scale1 = nn.Sequential(
            DepthwiseSeparableConv(32, 64, 3, stride=2, padding=1),
            DepthwiseSeparableConv(64, 64, 3, stride=1, padding=1),
        )
        self.scale2 = nn.Sequential(
            DepthwiseSeparableConv(32, 64, 5, stride=2, padding=2),
            DepthwiseSeparableConv(64, 64, 5, stride=1, padding=2),
        )
        self.merge_se = nn.ModuleDict({
            'fc': nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(128, 8),
                nn.ReLU(),
                nn.Linear(8, 128),
                nn.Sigmoid(),
            )
        })
        self.proj = nn.Sequential(
            nn.Conv2d(128, 128, 1, bias=False),
            nn.BatchNorm2d(128),
        )
        self.shallow = nn.Sequential(
            DepthwiseSeparableConv(32, 128, 1, stride=1, padding=0),
        )

    def forward(self, x):
        stem = F.relu(self.stem(x))
        s1 = F.relu(self.scale1(stem))
        s2 = F.relu(self.scale2(stem))
        merged = torch.cat([s1, s2], dim=1)
        se = self.merge_se['fc'](merged).unsqueeze(-1).unsqueeze(-1)
        merged = merged * se
        merged = F.relu(self.proj(merged))
        return merged, stem


class PatchSelector(nn.Module):
    def __init__(self, feat_dim=128, embed_dim=256, num_patches=49):
        super().__init__()
        self.saliency = nn.Conv2d(feat_dim, 1, 1)
        self.proj = nn.Linear(32768, embed_dim)  # 128 * 256 (16x16 feature map)
        self.pos_emb = nn.Embedding(num_patches, embed_dim)
        self.register_buffer('pos_ids', torch.arange(num_patches))

    def forward(self, x):
        B, C, H, W = x.shape
        sal = torch.sigmoid(self.saliency(x))
        x_weighted = x * sal
        tokens = x_weighted.flatten(2).transpose(1, 2)
        tokens = tokens.reshape(B, -1)
        tokens = self.proj(tokens).unsqueeze(1)
        pos = self.pos_emb(self.pos_ids).unsqueeze(0)
        N = min(tokens.shape[1], pos.shape[1])
        tokens = tokens[:, :N] + pos[:, :N]
        return tokens


class TransformerBlock(nn.Module):
    def __init__(self, dim=256, heads=8, mlp_ratio=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_dim, dim),
        )

    def forward(self, x):
        xn = self.norm1(x)
        x = x + self.attn(xn, xn, xn)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class LDASNTransformer(nn.Module):
    def __init__(self, dim=256, depth=4, heads=8):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B = x.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x[:, 0])


class ClassificationHead(nn.Module):
    def __init__(self, dim=256, num_classes=7):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        return self.fc(x) / self.temperature


class LDASN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.extractor = MultiScaleExtractor()
        self.selector = PatchSelector(feat_dim=128, embed_dim=256, num_patches=49)
        self.transformer = LDASNTransformer(dim=256, depth=4, heads=8)
        self.head = ClassificationHead(dim=256, num_classes=num_classes)

    def forward(self, x):
        features, _ = self.extractor(x)
        tokens = self.selector(features)
        cls_out = self.transformer(tokens)
        return self.head(cls_out)


# ─── Model Loading ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model(arch_key, model_path, num_classes):
    """Build architecture and load trained weights."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if arch_key == "Swin_T":
        model = LDASN(num_classes=num_classes)
    elif arch_key == "ConvNeXt_T":
        model = models.convnext_tiny(weights=None)
        in_f = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_f, num_classes)
    else:
        raise ValueError(f"Unknown architecture: {arch_key}")

    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device


def get_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def predict(model, image, device, class_names, img_size=224):
    """Run inference and return class probabilities."""
    tf = get_transform(img_size)
    img_tensor = tf(image).unsqueeze(0).to(device)

    start = time.time()
    with torch.no_grad():
        logits = model(img_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    inference_time = (time.time() - start) * 1000  # ms

    pred_idx = int(np.argmax(probs))
    pred_class = class_names[pred_idx]
    confidence = float(probs[pred_idx])

    return {
        "class": pred_class,
        "index": pred_idx,
        "confidence": confidence,
        "probabilities": {cn: float(probs[i]) for i, cn in enumerate(class_names)},
        "inference_time_ms": inference_time,
    }


# ─── AI Chatbot Logic (Google Gemini) ──────────────────────────────────────
SYSTEM_PROMPT = """You are Cotton Guard Assistant — an AI expert on cotton leaf diseases built into a disease detection app for Pakistani cotton farmers.

ABOUT THIS APP:
- Cotton Guard is a deep learning-based cotton leaf disease detection system
- Users upload a cotton leaf photo, select a dataset, and get instant diagnosis with confidence scores, disease info, and treatment recommendations

MODELS & DATASETS:
- SAR-CLD 2024 dataset (7 classes): Bacterial Blight, Curl Virus, Healthy Leaf, Herbicide Growth Damage, Leaf Hopper Jassids, Leaf Redding, Leaf Variegation. Best model: LDASN (Lightweight Dynamic Attention Selection Network) — a custom architecture with multi-scale feature extractor, saliency-based patch selector, and transformer blocks. Input size: 64×64. Accuracy: ~98.4%
- Cotton Leaf Disease dataset (4 classes): Bacterial Blight, Curl Virus, Fussarium Wilt, Healthy. Best model: ConvNeXt Tiny. Input size: 224×224. Accuracy: ~97.7%

TRAINING DETAILS:
- Both models use: 80/20 stratified split, Focal Loss with class weights, AdamW optimizer (lr=3e-4), Cosine Annealing LR, data augmentation (flips, rotations, color jitter), early stopping (patience=10), WeightedRandomSampler, ImageNet normalization
- Metrics: Balanced Accuracy, Weighted F1, MCC, Cohen's Kappa

DISEASE KNOWLEDGE:
- Bacterial Blight: Severity High. Angular water-soaked lesions turning brown. Treatment: copper-based bactericides, resistant varieties. Prevention: disease-free seeds, crop rotation.
- Curl Virus (CLCuV): Severity Very High. Transmitted by whiteflies, leaf curling, stunted growth. Treatment: control whiteflies (imidacloprid, acetamiprid). Prevention: resistant varieties, early sowing.
- Fussarium Wilt: Severity High. Soil-borne fungal disease blocking water vessels. Treatment: remove infected plants, soil solarization, Trichoderma biocontrol. Prevention: resistant varieties, crop rotation 3+ years.
- Herbicide Growth Damage: Severity Medium. Abnormal leaf growth from herbicide drift. Treatment: growth regulators, adequate irrigation. Prevention: proper spraying techniques, buffer zones.
- Leaf Hopper Jassids: Severity Medium-High. Jassids suck cell sap causing yellowing. Treatment: systemic insecticides (thiamethoxam). Prevention: resistant varieties, maintain natural predators.
- Leaf Redding: Severity Medium. Reddening due to nutrient deficiency (magnesium). Treatment: foliar magnesium sulphate. Prevention: soil testing, balanced NPK.
- Leaf Variegation: Severity Medium. Irregular color patches from viral infection. Treatment: remove affected plants, control vectors. Prevention: virus-free material, field hygiene.
- Healthy Leaf: No symptoms. Continue regular crop management.

YOUR RULES:
1. ONLY answer questions related to: cotton diseases, cotton farming, this app, its models, treatments, prevention, and cotton agriculture
2. If someone asks about anything unrelated (politics, general knowledge, coding, etc.), politely redirect them: "I'm specialized in cotton leaf diseases and this app. I can help you with disease identification, treatments, prevention tips, or how to use Cotton Guard."
3. Keep responses concise and farmer-friendly
4. You can respond in Urdu/Roman Urdu if the user writes in Urdu
5. Always be helpful and encouraging to farmers
6. Never make up information — stick to what you know about cotton diseases"""


def get_gemini_response(user_msg, chat_history):
    """Get AI response from Google Gemini API."""
    import requests
    import json

    api_key = st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        return "⚠️ Gemini API key not configured. Please add GEMINI_API_KEY to your Streamlit secrets."

    # Build conversation history for Gemini
    contents = []
    for msg in chat_history[-10:]:  # Last 10 messages for context
        role = "user" if msg["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})
    
    # Add current message
    contents.append({"role": "user", "parts": [{"text": user_msg}]})

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent?key={api_key}"
    
    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": contents,
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 500,
        }
    }

    try:
        response = requests.post(url, json=payload, timeout=15)
        data = response.json()
        
        if "candidates" in data and len(data["candidates"]) > 0:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        elif "error" in data:
            return f"⚠️ API Error: {data['error'].get('message', 'Unknown error')}"
        else:
            return "Sorry, I couldn't generate a response. Please try again."
    except requests.exceptions.Timeout:
        return "⚠️ Response timed out. Please try again."
    except Exception as e:
        return f"⚠️ Connection error. Please check your internet and try again."


# ─── App Header ────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1>🌿 Cotton Guard</h1>
    <p>Deep Learning Cotton Leaf Disease Detection System</p>
</div>
""", unsafe_allow_html=True)

# ─── Main Layout: Left (Detection) | Right (Chatbot) ──────────────────────
col_main, col_chat = st.columns([3, 2])

# ===== LEFT COLUMN — Detection =====
with col_main:
    # Dataset selection
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="glass-card-header">🗂️ Select Dataset & Model</div>', unsafe_allow_html=True)
    dataset_choice = st.selectbox(
        "Choose dataset",
        list(DATASET_INFO.keys()),
        label_visibility="collapsed",
    )
    ds = DATASET_INFO[dataset_choice]
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card">
            <div class="metric-label">Architecture</div>
            <div class="metric-value">{ds['architecture'].split('(')[0].strip()}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Classes</div>
            <div class="metric-value">{len(ds['classes'])}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Input Size</div>
            <div class="metric-value">{ds['img_size']}×{ds['img_size']}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Normalization</div>
            <div class="metric-value">ImageNet</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Image upload
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="glass-card-header">📷 Image Input</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload a cotton leaf image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption=uploaded_file.name, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Analyze button
    analyze_clicked = st.button("🔬 Analyze Leaf", use_container_width=True, type="primary")

    # ─── Results ───────────────────────────────────────────────────────────
    if analyze_clicked and uploaded_file is not None:
        try:
            with st.spinner("Loading model and analyzing..."):
                model, device = load_model(
                    ds["arch_key"], ds["model_file"], len(ds["classes"])
                )
                result = predict(model, image, device, ds["classes"], ds["img_size"])

            pred_class = result["class"]
            confidence = result["confidence"]
            is_healthy = pred_class in ["Healthy", "Healthy Leaf"]

            # Prediction card
            if is_healthy:
                box_class = "prediction-box"
                label_class = "prediction-label"
                icon = "✅"
            else:
                box_class = "disease-box"
                label_class = "disease-label"
                icon = "⚠️"

            st.markdown(f"""
            <div class="{box_class}">
                <div class="{label_class}">{icon} Prediction</div>
                <div class="prediction-name">{pred_class}</div>
                <div class="prediction-index">Class Index: {result['index']}</div>
            </div>
            """, unsafe_allow_html=True)

            # Confidence & Metrics
            res_col1, res_col2 = st.columns([1, 2])

            with res_col1:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown('<div class="glass-card-header">Confidence Score</div>', unsafe_allow_html=True)
                conf_color = "#4ade80" if confidence > 0.8 else "#facc15" if confidence > 0.5 else "#ef4444"
                st.markdown(f"""
                <div class="confidence-section">
                    <div class="confidence-pct" style="color:{conf_color}">{confidence*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(confidence)
                st.markdown('</div>', unsafe_allow_html=True)

            with res_col2:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown('<div class="glass-card-header">All Class Probabilities</div>', unsafe_allow_html=True)
                sorted_probs = sorted(result["probabilities"].items(), key=lambda x: -x[1])
                for cn, prob in sorted_probs:
                    bar_color = "#4ade80" if cn == pred_class else "#334155"
                    st.markdown(f"""
                    <div class="prob-item">
                        <div class="prob-name">{cn}</div>
                        <div class="prob-bar-bg">
                            <div class="prob-bar-fill" style="width:{prob*100:.1f}%;background:{bar_color}"></div>
                        </div>
                        <div class="prob-pct">{prob*100:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # Performance metrics
            st.markdown(f"""
            <div class="glass-card">
                <div class="glass-card-header">Performance Metrics</div>
                <div class="metric-row">
                    <div class="metric-card">
                        <div class="metric-label">Inference Time</div>
                        <div class="metric-value">{result['inference_time_ms']:.1f}ms</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Model</div>
                        <div class="metric-value">{ds['arch_key']}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Image Size</div>
                        <div class="metric-value">{image.size[0]}×{image.size[1]}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Dataset</div>
                        <div class="metric-value">{dataset_choice.split('—')[0].strip()}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Disease information
            if pred_class in DISEASE_INFO:
                info = DISEASE_INFO[pred_class]
                st.markdown(f"""
                <div class="glass-card">
                    <div class="glass-card-header">{'🌿 Crop Health Status' if is_healthy else '🔬 Disease Information & Treatment'}</div>
                </div>
                """, unsafe_allow_html=True)

                if not is_healthy:
                    st.markdown(f"""
                    <div class="info-card">
                        <h4>⚡ Severity: {info['severity']}</h4>
                        <p>{info['description']}</p>
                    </div>
                    <div class="info-card">
                        <h4>🔍 Symptoms</h4>
                        <p>{info['symptoms']}</p>
                    </div>
                    <div class="info-card">
                        <h4>💊 Recommended Treatment</h4>
                        <p>{info['treatment']}</p>
                    </div>
                    <div class="info-card">
                        <h4>🛡️ Prevention</h4>
                        <p>{info['prevention']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="info-card">
                        <h4>✅ Your cotton leaf looks healthy!</h4>
                        <p>{info['description']}</p>
                    </div>
                    <div class="info-card">
                        <h4>🌱 Maintenance Tips</h4>
                        <p>{info['prevention']}</p>
                    </div>
                    """, unsafe_allow_html=True)

        except FileNotFoundError:
            st.error(f"⚠️ Model file not found: `{ds['model_file']}`. Please ensure the model file is in the `models/` directory.")
        except Exception as e:
            st.error(f"Error during inference: {str(e)}")

    elif analyze_clicked and uploaded_file is None:
        st.warning("Please upload a cotton leaf image first.")


# ===== RIGHT COLUMN — Chatbot =====
with col_chat:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #10b981, #06b6d4);
                padding: 1rem 1.2rem; border-radius: 16px 16px 0 0;
                margin-top: 0;">
        <div style="font-weight: 700; color: white; font-size: 1rem;">🤖 Cotton Guard Assistant</div>
        <div style="color: rgba(255,255,255,0.8); font-size: 0.75rem;">Ask me about diseases, treatments & how this app works</div>
    </div>
    """, unsafe_allow_html=True)

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "assistant", "content": "Hello! 👋 I'm your Cotton Guard Assistant. I can help you with:\n\n"
             "🌿 Cotton disease information\n"
             "💊 Treatment recommendations\n"
             "🔬 How this app works\n\n"
             "What would you like to know?"}
        ]

    # Chat messages container
    chat_container = st.container(height=420)
    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "👤"):
                st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input("Ask about cotton diseases...", key="chat_input")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.spinner("Thinking..."):
            response = get_gemini_response(user_input, st.session_state.chat_history)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()
