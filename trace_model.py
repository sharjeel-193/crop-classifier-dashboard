import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from PIL import Image

@torch.no_grad()
def inference(model, img_path):
    classes = [
        "Bacterial_spot", 
        "Early_blight", 
        "Late_blight", 
        "Leaf_Mold",
        "Septoria_leaf_spot", 
        "Spider_mites_Two-spotted_spider_mite", 
        "Target_Spot", 
        "Tomato_Yellow_Leaf_Curl_Virus", 
        "Tomato_Yellow_Leaf_Curl_Virus", 
        "healthy", 
        "powdery_mildew"
        ]
    transform=T.Compose([
        T.Resize(size=(224, 224)),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img = Image.open(img_path).convert('RGB')
    img = transform(img).unsqueeze(0)
    pt_result = model(img)
    return classes[pt_result.argmax().item()]


def load_model(path):
    model = torchvision.models.mobilenet_v3_small()
    in_features = model._modules['classifier'][-1].in_features
    out_features = 11
    model._modules['classifier'][-1] = nn.Linear(in_features, out_features, bias=True)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model = nn.Sequential(
        model,
        nn.Softmax(1)
    )
    model.eval()
    return model

model = load_model("model.pt")

# imgs = [
# "bacteria.jpeg",
# "healthy.jpeg",
# "septoria.jpeg"
# ]
# for img in imgs:
#     print(inference(model, img))
# print("=================================================")
# example = torch.rand(1, 3, 224, 224)
# traced_script_module = torch.jit.trace(model, example)
# for img in imgs:
#     print(inference(traced_script_module, img))
# # optimized_traced_model = optimize_for_mobile(traced_script_module)
# traced_script_module._save_for_lite_interpreter("app/src/main/assets/model11.pt")
# print(model)