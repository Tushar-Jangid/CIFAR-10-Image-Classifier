from flask import Flask, request,jsonify
import torch
import io
from utils import preprocess_image_bytes, CIFAR10_CLASSES
from model import  CIFARResNet18
import os

app = Flask(__name__)


Model_Path = os.environ.get("Model_Path","models/resent.pth")
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(path=Model_Path,device=device):
    model = CIFARResNet18(num_classes=10)
    ckpt  = torch.load(path,map_location=device)
    model.load_state_dict(ckpt['model_state'] if 'model_state' in ckpt else ckpt)
    model.to(device).eval()
    return model
model = load_model()

@app.route('/heath',method=['GET'])
def heath():
    return jsonify({"Status":"ok"})

@app.route("/predict",method=['POST'])
def prdict():
    if "file" not in request.files:
        return jsonify({"error": "no file uploaded"}),400
    file = request.files['file']
    img_bytes = file.read()
    try:
        tensor = preprocess_image_bytes(img_bytes)
    except Exception as e:
        return jsonify({"error":"invaild image","detail":str(e)}),400
    
    tensor = tensor.to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs,dim=1).cpu().numpy().tolist()[0]
        pred_idx = int(torch.argmax(outputs,dim=1).cpu().item())

    return jsonify({
        "predicted_class": CIFAR10_CLASSES[pred_idx],
        "predicted_index": pred_idx,
        "probabilities": {CIFAR10_CLASSES[i]: float(probs[i]) for i in range(len(probs))}
    })
if __name__ == "__main__":
    app.run(host='0.0.0.0',post=5000,debug=False)