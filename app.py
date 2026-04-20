"""
Cotton Guard — Cotton Leaf Disease Detection System
=====================================================
Deep learning application for Pakistani cotton farmers.
Datasets: SAR-CLD 2024 (7 classes) → LDASN | Cotton Leaf Disease (4 classes) → ConvNeXt-T
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import time

st.set_page_config(page_title="Cotton Guard — Leaf Disease Detection", page_icon="🍃", layout="wide", initial_sidebar_state="expanded")

# ─── Warm Earthy Theme CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=Nunito:wght@300;400;500;600;700;800&family=Fira+Code:wght@400;500&display=swap');
.stApp { background: #e8efe2; font-family: 'Nunito', sans-serif; }
.app-header { text-align:center; padding:1.5rem 0 0.5rem; }
.app-header h1 { font-family:'DM Serif Display',serif; font-size:2.6rem; color:#2d5016; margin:0; }
.app-header .subtitle { color:#7a8b6e; font-size:0.95rem; margin-top:0.2rem; }
.header-divider { width:80px; height:3px; background:linear-gradient(90deg,#8b6914,#2d5016,#8b6914); margin:0.8rem auto 1.5rem; border-radius:2px; }
.earth-card { background:#d9e4cf; border:1px solid #b8c9a8; border-radius:14px; padding:1.4rem; margin-bottom:1rem; box-shadow:0 2px 12px rgba(45,80,22,0.08); }
.earth-card-header { font-family:'Fira Code',monospace; font-size:0.68rem; font-weight:500; color:#8b6914; text-transform:uppercase; letter-spacing:2.5px; margin-bottom:0.8rem; padding-bottom:0.5rem; border-bottom:1px solid #c5d4b5; }
.prediction-box { background:linear-gradient(135deg,#c8ddb8,#b8d0a5); border:1px solid #8baf72; border-left:4px solid #2d5016; border-radius:12px; padding:1.2rem 1.5rem; margin:0.8rem 0; }
.prediction-label { font-size:0.68rem; font-weight:700; color:#2d5016; text-transform:uppercase; letter-spacing:2.5px; }
.prediction-name { font-family:'DM Serif Display',serif; font-size:1.8rem; color:#1a3a0a; margin:0.3rem 0 0.1rem; }
.prediction-index { font-size:0.82rem; color:#5a6650; }
.disease-box { background:linear-gradient(135deg,#efe0c0,#e8d5a8); border:1px solid #d4a843; border-left:4px solid #8b6914; border-radius:12px; padding:1.2rem 1.5rem; margin:0.8rem 0; }
.disease-label { font-size:0.68rem; font-weight:700; color:#8b6914; text-transform:uppercase; letter-spacing:2.5px; }
.confidence-section { text-align:center; padding:0.8rem 0; }
.confidence-pct { font-family:'DM Serif Display',serif; font-size:2.8rem; line-height:1; }
.metric-row { display:flex; gap:0.7rem; flex-wrap:wrap; margin:0.8rem 0; }
.metric-card { flex:1; min-width:110px; background:#cddabe; border:1px solid #b8c9a8; border-radius:10px; padding:0.7rem 0.5rem; text-align:center; }
.metric-card .metric-label { font-size:0.6rem; color:#8b6914; text-transform:uppercase; letter-spacing:1.5px; font-weight:600; }
.metric-card .metric-value { font-size:0.95rem; font-weight:700; color:#2d3a1e; margin-top:0.15rem; font-family:'Fira Code',monospace; }
.prob-item { display:flex; align-items:center; margin:0.45rem 0; gap:0.7rem; }
.prob-name { width:170px; font-size:0.8rem; color:#3a4a30; font-weight:600; text-align:right; flex-shrink:0; }
.prob-bar-bg { flex:1; height:10px; background:#c5d4b5; border-radius:5px; overflow:hidden; }
.prob-bar-fill { height:100%; border-radius:5px; transition:width 0.5s ease; }
.prob-pct { width:48px; font-size:0.75rem; color:#5a6650; font-family:'Fira Code',monospace; text-align:right; flex-shrink:0; }
.info-card { background:#cddabe; border:1px solid #b8c9a8; border-radius:10px; padding:1rem 1.2rem; margin:0.5rem 0; }
.info-card h4 { color:#2d5016; font-size:0.88rem; margin:0 0 0.35rem 0; font-weight:700; }
.info-card p { color:#3a4a30; font-size:0.82rem; margin:0; line-height:1.6; }
.info-card-warn h4 { color:#8b6914; }
div[data-testid="stFileUploader"] { background:#d9e4cf; border:2px dashed #a8b898; border-radius:14px; padding:1rem; }
div[data-testid="stFileUploader"]:hover { border-color:#8b6914; }
section[data-testid="stSidebar"] { background:#2d3a1e !important; }
section[data-testid="stSidebar"] .stMarkdown p, section[data-testid="stSidebar"] .stMarkdown li { color:#d4dccb !important; }
div[data-testid="stSelectbox"] > div > div { background:#d9e4cf !important; border-color:#b8c9a8 !important; }
div[data-testid="stMainBlockContainer"] { background:#e8efe2 !important; }
.stSelectbox label { color:#5a6650 !important; font-family:'Fira Code',monospace !important; font-size:0.68rem !important; text-transform:uppercase; letter-spacing:2px; }
.stButton > button[kind="primary"] { background:linear-gradient(135deg,#2d5016,#3d6b20) !important; color:white !important; border:none !important; border-radius:10px !important; font-weight:700 !important; font-size:1rem !important; }
.stButton > button[kind="primary"]:hover { background:linear-gradient(135deg,#3d6b20,#4a8028) !important; }
.stProgress > div > div { background-color:#2d5016 !important; }
#MainMenu {visibility:hidden;} footer {visibility:hidden;} .stDeployButton {display:none;}
</style>
""", unsafe_allow_html=True)

# ─── Constants ─────────────────────────────────────────────────────────────
SAR_CLD_CLASSES = ["Bacterial Blight","Curl Virus","Healthy Leaf","Herbicide Growth Damage","Leaf Hopper Jassids","Leaf Redding","Leaf Variegation"]
COTTON_LEAF_CLASSES = ["Bacterial Blight","Curl Virus","Fussarium Wilt","Healthy"]
DATASET_INFO = {
    "SAR-CLD 2024 — 7 Classes": {"classes":SAR_CLD_CLASSES,"model_file":"models/swin_t_best.pt","architecture":"LDASN (Lightweight Dynamic Attention)","arch_key":"Swin_T","img_size":64},
    "Cotton Leaf Disease — 4 Classes": {"classes":COTTON_LEAF_CLASSES,"model_file":"models/convnext_t_best.pt","architecture":"ConvNeXt Tiny (ConvNeXt-T)","arch_key":"ConvNeXt_T","img_size":224},
}
DISEASE_INFO = {
    "Bacterial Blight": {"severity":"High","icon":"🔴","description":"Angular water-soaked lesions on leaves that turn brown. Causes defoliation and boll rot.","symptoms":"Water-soaked angular spots, blackening of veins, premature defoliation.","treatment":"Use copper-based bactericides. Plant resistant varieties. Remove and destroy infected debris.","prevention":"Use certified disease-free seeds, crop rotation with non-host crops, avoid overhead irrigation."},
    "Curl Virus": {"severity":"Very High","icon":"🔴","description":"Transmitted by whiteflies, causes upward or downward curling of leaves, stunted growth, and severe yield loss.","symptoms":"Leaf curling, thickened veins, enation (leaf-like outgrowths), stunted plants, reduced boll formation.","treatment":"Control whitefly vectors with insecticides (imidacloprid, acetamiprid). Remove infected plants early. Use sticky traps.","prevention":"Plant resistant varieties (BT cotton with CLCuV tolerance), early sowing, maintain field hygiene."},
    "Healthy Leaf": {"severity":"None","icon":"🟢","description":"The leaf appears healthy with no visible signs of disease or pest damage.","symptoms":"No symptoms — uniform green color, normal leaf shape and size.","treatment":"No treatment needed. Continue regular crop management.","prevention":"Maintain balanced nutrition, proper irrigation scheduling, and regular scouting."},
    "Healthy": {"severity":"None","icon":"🟢","description":"The leaf appears healthy with no visible signs of disease or pest damage.","symptoms":"No symptoms — uniform green color, normal leaf shape and size.","treatment":"No treatment needed. Continue regular crop management.","prevention":"Maintain balanced nutrition, proper irrigation scheduling, and regular scouting."},
    "Herbicide Growth Damage": {"severity":"Medium","icon":"🟡","description":"Damage from herbicide drift or misapplication, resulting in abnormal leaf growth.","symptoms":"Cupped or strapped leaves, abnormal growth, epinasty, chlorosis.","treatment":"Foliar application of growth regulators. Provide adequate irrigation and nutrition.","prevention":"Proper herbicide application techniques, avoid spraying on windy days, calibrate sprayers."},
    "Leaf Hopper Jassids": {"severity":"Medium-High","icon":"🟠","description":"Jassids suck cell sap from leaves causing yellowing and curling of leaf margins.","symptoms":"Yellowing of leaf margins, downward curling, hopper burn in severe cases.","treatment":"Apply systemic insecticides (thiamethoxam, imidacloprid). Use neem-based sprays.","prevention":"Use resistant varieties, intercropping, maintain natural predators."},
    "Leaf Redding": {"severity":"Medium","icon":"🟡","description":"Reddening of leaves due to nutrient deficiency (often magnesium) or physiological stress.","symptoms":"Reddish-purple discoloration, starting from lower leaves and moving upward.","treatment":"Foliar application of magnesium sulphate. Correct nutrient imbalances.","prevention":"Regular soil testing, balanced NPK application."},
    "Leaf Variegation": {"severity":"Medium","icon":"🟡","description":"Irregular patches of different colors on leaves, often caused by viral infections.","symptoms":"Mosaic patterns, irregular light and dark green patches, sometimes yellow streaks.","treatment":"Remove severely affected plants. Control insect vectors.","prevention":"Use virus-free planting material, control aphid and whitefly vectors."},
    "Fussarium Wilt": {"severity":"High","icon":"🔴","description":"Soil-borne fungal disease that blocks water-conducting vessels, causing wilting and death.","symptoms":"Yellowing on one side, wilting despite adequate moisture, brown vascular tissue.","treatment":"Remove and destroy infected plants. Soil solarization. Trichoderma biocontrol.","prevention":"Use resistant varieties, long crop rotation (3+ years), avoid waterlogging."},
}

# ─── LDASN Architecture ───────────────────────────────────────────────────
class DepthwiseSeparableConv(nn.Module):
    def __init__(s,ic,oc,k,st=1,p=0):
        super().__init__(); s.dw=nn.Conv2d(ic,ic,k,st,p,groups=ic,bias=False); s.pw=nn.Conv2d(ic,oc,1,bias=False); s.bn=nn.BatchNorm2d(oc)
    def forward(s,x): return s.bn(s.pw(s.dw(x)))
class MultiScaleExtractor(nn.Module):
    def __init__(s):
        super().__init__(); s.stem=nn.Sequential(nn.Conv2d(3,32,3,stride=2,padding=1,bias=False),nn.BatchNorm2d(32))
        s.scale1=nn.Sequential(DepthwiseSeparableConv(32,64,3,2,1),DepthwiseSeparableConv(64,64,3,1,1))
        s.scale2=nn.Sequential(DepthwiseSeparableConv(32,64,5,2,2),DepthwiseSeparableConv(64,64,5,1,2))
        s.merge_se=nn.ModuleDict({'fc':nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(128,8),nn.ReLU(),nn.Linear(8,128),nn.Sigmoid())})
        s.proj=nn.Sequential(nn.Conv2d(128,128,1,bias=False),nn.BatchNorm2d(128)); s.shallow=nn.Sequential(DepthwiseSeparableConv(32,128,1,1,0))
    def forward(s,x):
        stem=F.relu(s.stem(x)); s1=F.relu(s.scale1(stem)); s2=F.relu(s.scale2(stem)); m=torch.cat([s1,s2],1)
        return F.relu(s.proj(m*s.merge_se['fc'](m).unsqueeze(-1).unsqueeze(-1))), stem
class PatchSelector(nn.Module):
    def __init__(s,fd=128,ed=256,np_=49):
        super().__init__(); s.saliency=nn.Conv2d(fd,1,1); s.proj=nn.Linear(32768,ed); s.pos_emb=nn.Embedding(np_,ed); s.register_buffer('pos_ids',torch.arange(np_))
    def forward(s,x):
        B=x.shape[0]; t=(x*torch.sigmoid(s.saliency(x))).flatten(2).transpose(1,2).reshape(B,-1)
        t=s.proj(t).unsqueeze(1); p=s.pos_emb(s.pos_ids).unsqueeze(0); n=min(t.shape[1],p.shape[1]); return t[:,:n]+p[:,:n]
class TransformerBlock(nn.Module):
    def __init__(s,d=256,h=8,r=2):
        super().__init__(); s.norm1=nn.LayerNorm(d); s.attn=nn.MultiheadAttention(d,h,batch_first=True); s.norm2=nn.LayerNorm(d); s.mlp=nn.Sequential(nn.Linear(d,d*r),nn.GELU(),nn.Dropout(0.1),nn.Linear(d*r,d))
    def forward(s,x): xn=s.norm1(x); x=x+s.attn(xn,xn,xn)[0]; return x+s.mlp(s.norm2(x))
class LDASNTransformer(nn.Module):
    def __init__(s,d=256,dp=4,h=8):
        super().__init__(); s.cls_token=nn.Parameter(torch.randn(1,1,d)); s.blocks=nn.ModuleList([TransformerBlock(d,h) for _ in range(dp)]); s.norm=nn.LayerNorm(d)
    def forward(s,x):
        x=torch.cat([s.cls_token.expand(x.shape[0],-1,-1),x],1)
        for b in s.blocks: x=b(x)
        return s.norm(x[:,0])
class ClassificationHead(nn.Module):
    def __init__(s,d=256,nc=7): super().__init__(); s.temperature=nn.Parameter(torch.ones(1)); s.fc=nn.Linear(d,nc)
    def forward(s,x): return s.fc(x)/s.temperature
class LDASN(nn.Module):
    def __init__(s,nc=7):
        super().__init__(); s.extractor=MultiScaleExtractor(); s.selector=PatchSelector(128,256,49); s.transformer=LDASNTransformer(256,4,8); s.head=ClassificationHead(256,nc)
    def forward(s,x): f,_=s.extractor(x); return s.head(s.transformer(s.selector(f)))

# ─── Model Loading ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model(arch_key, model_path, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if arch_key == "Swin_T": model = LDASN(num_classes)
    else: model = models.convnext_tiny(weights=None); model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False)); model.to(device).eval(); return model, device

def predict(model, image, device, class_names, img_size=224):
    tf = transforms.Compose([transforms.Resize((img_size,img_size)),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    start = time.time()
    with torch.no_grad(): probs = F.softmax(model(tf(image).unsqueeze(0).to(device)),1).cpu().numpy()[0]
    idx = int(np.argmax(probs))
    return {"class":class_names[idx],"index":idx,"confidence":float(probs[idx]),"probabilities":{cn:float(probs[i]) for i,cn in enumerate(class_names)},"inference_time_ms":(time.time()-start)*1000}

# ─── AI Chatbot (Groq) ────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are Cotton Guard Assistant — an AI expert on cotton leaf diseases for Pakistani cotton farmers.
APP: Cotton Guard detects cotton leaf diseases via deep learning. Users upload leaf photos for instant diagnosis.
MODELS: SAR-CLD 2024 (7 classes, LDASN, 64x64, ~98.4% acc) | Cotton Leaf Disease (4 classes, ConvNeXt-T, 224x224, ~97.7% acc).
TRAINING: 80/20 split, Focal Loss, AdamW, Cosine LR, augmentation, early stopping.
DISEASES: Bacterial Blight (High), Curl Virus/CLCuV (Very High), Fussarium Wilt (High), Herbicide Damage (Medium), Jassids (Medium-High), Leaf Redding (Medium), Leaf Variegation (Medium).
RULES: Only answer about cotton diseases/farming/this app. Redirect unrelated questions politely. Be concise, farmer-friendly. ALWAYS respond in English by default. Only respond in Urdu/Roman Urdu if the user explicitly asks you to write in Urdu or writes their message in Urdu. Never make up info."""

def get_ai_response(user_msg, chat_history):
    import requests
    api_key = st.secrets.get("GROQ_API_KEY", "")
    if not api_key: return "⚠️ Please add GROQ_API_KEY to Streamlit secrets."
    messages = [{"role":"system","content":SYSTEM_PROMPT}] + [{"role":m["role"],"content":m["content"]} for m in chat_history[-10:]] + [{"role":"user","content":user_msg}]
    try:
        r = requests.post("https://api.groq.com/openai/v1/chat/completions", json={"model":"llama-3.3-70b-versatile","messages":messages,"temperature":0.7,"max_tokens":500},
            headers={"Authorization":f"Bearer {api_key}","Content-Type":"application/json"}, timeout=15).json()
        return r["choices"][0]["message"]["content"] if "choices" in r else f"⚠️ {r.get('error',{}).get('message','Error')}"
    except: return "⚠️ Connection error. Please try again."

# ╔═══════════════════════ SIDEBAR — CHATBOT ═══════════════════════════════╗
with st.sidebar:
    st.markdown("""<div style="text-align:center;padding:0.5rem 0 0.8rem;"><span style="font-size:2rem;">🍃</span>
    <h2 style="font-family:'DM Serif Display',serif;color:#e8f0e0;margin:0.2rem 0 0;">Cotton Guard</h2>
    <p style="color:#a8b89e;font-size:0.78rem;margin:0;">AI Assistant</p></div>
    <hr style="border:none;border-top:1px solid #3d4e2e;margin:0 0 1rem;">""", unsafe_allow_html=True)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{"role":"assistant","content":"Assalam o Alaikum! 👋\n\nI'm your Cotton Guard Assistant. I can help with:\n\n🌿 Disease identification\n💊 Treatment advice\n🔬 How this app works\n\nAsk me anything about cotton crops!"}]
    chat_box = st.container(height=400)
    with chat_box:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"], avatar="🌿" if msg["role"]=="assistant" else "👤"):
                st.markdown(msg["content"])
    user_input = st.chat_input("Ask about cotton diseases...", key="chat_input")
    if user_input:
        st.session_state.chat_history.append({"role":"user","content":user_input})
        with st.spinner("Thinking..."): response = get_ai_response(user_input, st.session_state.chat_history)
        st.session_state.chat_history.append({"role":"assistant","content":response}); st.rerun()

# ╔═══════════════════════ MAIN AREA ═══════════════════════════════════════╗
st.markdown("""<div class="app-header"><h1>🍃 Cotton Guard</h1><p class="subtitle">Deep Learning Cotton Leaf Disease Detection for Farmers</p></div><div class="header-divider"></div>""", unsafe_allow_html=True)

# Dataset Selection
st.markdown('<div class="earth-card"><div class="earth-card-header">📋 Select Dataset & Model</div>', unsafe_allow_html=True)
dataset_choice = st.selectbox("Dataset", list(DATASET_INFO.keys()), label_visibility="collapsed")
ds = DATASET_INFO[dataset_choice]
st.markdown(f"""<div class="metric-row">
<div class="metric-card"><div class="metric-label">Architecture</div><div class="metric-value">{ds['architecture'].split('(')[0].strip()}</div></div>
<div class="metric-card"><div class="metric-label">Classes</div><div class="metric-value">{len(ds['classes'])}</div></div>
<div class="metric-card"><div class="metric-label">Input Size</div><div class="metric-value">{ds['img_size']}×{ds['img_size']}</div></div>
<div class="metric-card"><div class="metric-label">Normalization</div><div class="metric-value">ImageNet</div></div>
</div></div>""", unsafe_allow_html=True)

# Upload + Preview
col_up, col_prev = st.columns([1,1])
with col_up:
    st.markdown('<div class="earth-card"><div class="earth-card-header">📷 Upload Cotton Leaf</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload", type=["jpg","jpeg","png","bmp","webp"], label_visibility="collapsed")
    st.markdown('<p style="color:#7a8b6e;font-size:0.75rem;text-align:center;margin-top:0.5rem;">Upload a clear close-up photo of a single cotton leaf</p></div>', unsafe_allow_html=True)
with col_prev:
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.markdown('<div class="earth-card"><div class="earth-card-header">🖼️ Preview</div>', unsafe_allow_html=True)
        st.image(image, caption=uploaded_file.name, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Analyze
analyze = st.button("🔬  Analyze Leaf", use_container_width=True, type="primary")

if analyze and uploaded_file:
    try:
        with st.spinner("🌿 Analyzing your cotton leaf..."):
            model, device = load_model(ds["arch_key"], ds["model_file"], len(ds["classes"]))
            result = predict(model, image, device, ds["classes"], ds["img_size"])
        pc, conf = result["class"], result["confidence"]
        healthy = pc in ["Healthy","Healthy Leaf"]
        bc = "prediction-box" if healthy else "disease-box"
        lc = "prediction-label" if healthy else "disease-label"
        ic = "✅" if healthy else "⚠️"
        st.markdown(f'<div class="{bc}"><div class="{lc}">{ic} Prediction Result</div><div class="prediction-name">{pc}</div><div class="prediction-index">Class Index: {result["index"]} · Confidence: {conf*100:.1f}%</div></div>', unsafe_allow_html=True)
        c1, c2 = st.columns([1,2])
        with c1:
            cc = "#2d5016" if conf>0.8 else "#8b6914" if conf>0.5 else "#a83232"
            st.markdown(f'<div class="earth-card"><div class="earth-card-header">Confidence</div><div class="confidence-section"><div class="confidence-pct" style="color:{cc}">{conf*100:.1f}%</div></div>', unsafe_allow_html=True)
            st.progress(conf); st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="earth-card"><div class="earth-card-header">Class Probabilities</div>', unsafe_allow_html=True)
            for cn,p in sorted(result["probabilities"].items(), key=lambda x:-x[1]):
                bc2 = "#2d5016" if cn==pc else "#8b9e78"
                st.markdown(f'<div class="prob-item"><div class="prob-name">{cn}</div><div class="prob-bar-bg"><div class="prob-bar-fill" style="width:{p*100:.1f}%;background:{bc2}"></div></div><div class="prob-pct">{p*100:.1f}%</div></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="earth-card"><div class="earth-card-header">⚡ Performance</div><div class="metric-row"><div class="metric-card"><div class="metric-label">Inference</div><div class="metric-value">{result["inference_time_ms"]:.1f}ms</div></div><div class="metric-card"><div class="metric-label">Model</div><div class="metric-value">{ds["arch_key"]}</div></div><div class="metric-card"><div class="metric-label">Original</div><div class="metric-value">{image.size[0]}×{image.size[1]}</div></div><div class="metric-card"><div class="metric-label">Dataset</div><div class="metric-value">{dataset_choice.split("—")[0].strip()}</div></div></div></div>', unsafe_allow_html=True)
        if pc in DISEASE_INFO:
            info = DISEASE_INFO[pc]
            if not healthy:
                st.markdown(f'<div class="earth-card"><div class="earth-card-header">🔬 Disease Information & Treatment</div></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="info-card info-card-warn"><h4>{info.get("icon","")} Severity: {info["severity"]}</h4><p>{info["description"]}</p></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="info-card"><h4>🔍 Symptoms</h4><p>{info["symptoms"]}</p></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="info-card"><h4>💊 Treatment</h4><p>{info["treatment"]}</p></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="info-card"><h4>🛡️ Prevention</h4><p>{info["prevention"]}</p></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="info-card"><h4>✅ Your cotton leaf looks healthy!</h4><p>{info["description"]}</p></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="info-card"><h4>🌱 Maintenance Tips</h4><p>{info["prevention"]}</p></div>', unsafe_allow_html=True)
    except FileNotFoundError: st.error(f"⚠️ Model not found: `{ds['model_file']}`")
    except Exception as e: st.error(f"Error: {str(e)}")
elif analyze and not uploaded_file:
    st.warning("☝️ Please upload a cotton leaf image first.")
