from flask import Flask, render_template, request
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import io
import base64
import cv2

app = Flask(__name__)

# -------------------------------
# MODEL DEFINITION
# -------------------------------
class SkinCNN(nn.Module):
    def __init__(self):
        super(SkinCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 7)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load trained model
model = SkinCNN()
model.load_state_dict(torch.load("skin_model.pth", map_location="cpu"))
model.eval()

# -------------------------------
# CLASS LABELS (MULTILINGUAL)
# -------------------------------
class_names = {
    'en': {
        0: 'Actinic keratoses (Solar Lentigos / Seborrheic Keratoses / Bowen’s disease)',
        1: 'Basal cell carcinoma',
        2: 'Benign keratosis-like lesions',
        3: 'Dermatofibroma',
        4: 'Melanocytic nevi',
        5: 'Pyogenic granulomas and hemorrhage (Vascular lesions)',
        6: 'Melanoma'
    },
    'es': {
        0: 'Queratosis actínica (Lentigos solares / Queratosis seborreicas / Enfermedad de Bowen)',
        1: 'Carcinoma basocelular',
        2: 'Lesiones queratósicas benignas',
        3: 'Dermatofibroma',
        4: 'Nevus melanocítico',
        5: 'Granulomas piógenos y hemorragia (Lesiones vasculares)',
        6: 'Melanoma'
    },
    'fr': {
        0: 'Kératose actinique (Lentigines solaires / Kératoses séborrhéiques / Maladie de Bowen)',
        1: 'Épithélioma basocellulaire',
        2: 'Lésions kératosiques bénignes',
        3: 'Dermatofibrome',
        4: 'Naevus mélanocytaire',
        5: 'Granulomes pyogéniques et hémorragie (Lésions vasculaires)',
        6: 'Mélanome'
    }
}

# -------------------------------
# DANGER LEVELS + MEDICAL ADVICE
# -------------------------------
danger_info = {
    'en': {
        0: {'danger': 'Medium', 'advice': 'Precancerous; 5–10% may progress. Seek medical attention if it changes.'},
        1: {'danger': 'Medium', 'advice': 'Usually slow-growing; treat early. Consult a dermatologist promptly.'},
        2: {'danger': 'Low', 'advice': 'Harmless. Only check if rapidly growing or bleeding.'},
        3: {'danger': 'Low', 'advice': 'Benign. See a doctor if painful or changing.'},
        4: {'danger': 'Low', 'advice': 'Harmless but monitor moles for ABCDE changes.'},
        5: {'danger': 'Low', 'advice': 'Benign vascular lesion. Check if bleeding or infected.'},
        6: {'danger': 'High', 'advice': 'Very dangerous; early treatment is critical. See a doctor immediately.'}
    },
    'es': {
        0: {'danger': 'Medio', 'advice': 'Precanceroso; 5–10% de riesgo. Consulte si cambia.'},
        1: {'danger': 'Medio', 'advice': 'De crecimiento lento. Consulte pronto.'},
        2: {'danger': 'Bajo', 'advice': 'Inofensivo. Revise si crece rápido.'},
        3: {'danger': 'Bajo', 'advice': 'Benigno. Consulte si duele o cambia.'},
        4: {'danger': 'Bajo', 'advice': 'Monitorizar cambios ABCDE.'},
        5: {'danger': 'Bajo', 'advice': 'Lesión vascular benigna. Revisar si sangra.'},
        6: {'danger': 'Alto', 'advice': 'Muy peligroso; tratamiento urgente.'}
    },
    'fr': {
        0: {'danger': 'Moyen', 'advice': 'Précancéreux; surveiller les changements.'},
        1: {'danger': 'Moyen', 'advice': 'Croissance lente; consulter rapidement.'},
        2: {'danger': 'Faible', 'advice': 'Inoffensif. Vérifier si croissance rapide.'},
        3: {'danger': 'Faible', 'advice': 'Bénin. Vérifier si douloureux.'},
        4: {'danger': 'Faible', 'advice': 'Surveiller ABCDE.'},
        5: {'danger': 'Faible', 'advice': 'Lésion vasculaire bénigne.'},
        6: {'danger': 'Élevé', 'advice': 'Très dangereux; attention urgente.'}
    }
}

# -------------------------------
# GRAD-CAM IMPLEMENTATION
# -------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, input, output):
        self.gradients = output[0]

    def __call__(self, x, class_idx):
        logit = self.model(x)
        score = logit[:, class_idx]

        self.model.zero_grad()
        score.backward()

        gradients = self.gradients[0]
        activations = self.activations[0]

        weights = gradients.mean(dim=[1, 2], keepdim=True)
        cam = (weights * activations).sum(dim=0).cpu().detach().numpy()

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (28, 28))
        cam = cam / cam.max()

        return cam

# -------------------------------
# MAIN PAGE
# -------------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        lang = request.form.get('language', 'en')

        if file:
            # Read & preprocess image
            img = Image.open(file.stream)
            orig_img = img.copy()
            img = img.resize((28, 28))

            if img.mode != 'RGB':
                img = img.convert('RGB')

            arr = np.array(img).astype(np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

            # Prediction
            with torch.no_grad():
                output = model(tensor)
                probs = F.softmax(output, dim=1)[0] * 100

                top_probs, top_idxs = probs.topk(3)
                pred_idx = top_idxs[0].item()
                pred_name = class_names[lang][pred_idx]

                confidences = {
                    class_names[lang][i.item()]: f"{p.item():.2f}%"
                    for i, p in zip(top_idxs, top_probs)
                }

            # Danger & advice
            info = danger_info[lang][pred_idx]
            danger_key = danger_info['en'][pred_idx]['danger'].lower()

            # Grad-CAM
            gradcam = GradCAM(model, model.conv2)
            cam = gradcam(tensor, pred_idx)

            # Raw heatmap
            cam_color = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_JET)
            cam_color = cv2.cvtColor(cam_color, cv2.COLOR_BGR2RGB)
            raw_hm = Image.fromarray(cam_color)
            buf_raw = io.BytesIO()
            raw_hm.save(buf_raw, format="PNG")
            raw_heatmap = base64.b64encode(buf_raw.getvalue()).decode()

            # Overlay
            overlay = cv2.addWeighted(np.array(img), 0.5, cam_color, 0.5, 0)
            over_pil = Image.fromarray(overlay)
            buf = io.BytesIO()
            over_pil.save(buf, format="PNG")
            heatmap_base64 = base64.b64encode(buf.getvalue()).decode()

            # Original image
            buf2 = io.BytesIO()
            orig_img.save(buf2, format="PNG")
            orig_base64 = base64.b64encode(buf2.getvalue()).decode()

            return render_template(
                "result.html",
                pred_name=pred_name,
                confidences=confidences,
                danger_key=danger_key,
                danger=info['danger'],
                advice=info['advice'],
                heatmap=heatmap_base64,
                raw_heatmap=raw_heatmap,
                original=orig_base64,
                lang=lang
            )

    return render_template("index.html")

from flask import jsonify

@app.route('/get_dermatologists')
def get_dermatologists():
    query = request.args.get('query', '')

    # If you have a Google Places API key, use it here.
    # Since you don't want sample data, return empty unless real data is fetched.

    return jsonify({"results": []})
# -------------------------------
# RUN APP
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
