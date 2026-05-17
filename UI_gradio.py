import gradio as gr
import torch
import numpy as np
import cv2
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 256

def load_model():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1
    ).to(DEVICE)
    model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
    model.eval()
    return model

model = load_model()

transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

def predict(image_rgb):
    aug = transform(image=image_rgb)
    x = aug["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(x)
        pred = torch.sigmoid(pred)
        mask = (pred > 0.5).float().cpu().numpy()[0, 0]
    return mask

def make_overlay(image_rgb, mask):
    h, w = image_rgb.shape[:2]
    mask_resized = cv2.resize(mask.astype(np.uint8), (w, h))
    overlay = image_rgb.copy()
    overlay[mask_resized == 1] = [255, 0, 0] # Выделяем красным цветом в формате RGB
    result = cv2.addWeighted(image_rgb, 0.7, overlay, 0.3, 0)
    return result

def process_shelf_image(image):
    if image is None:
        return None, "Пожалуйста, загрузите изображение."
    
    mask = predict(image)
    result = make_overlay(image, mask)
    
    empty_percent = (mask.sum() / mask.size) * 100
    
    if empty_percent > 15:
        status_text = f"🔴 НУЖНО ПОПОЛНЕНИЕ ПОЛКИ!\nПустоты составляют: {empty_percent:.2f}%"
    else:
        status_text = f"🟢 Полка заполнена.\nПустоты составляют: {empty_percent:.2f}%"
        
    return result, status_text

demo = gr.Interface(
    fn=process_shelf_image,
    inputs=gr.Image(type="numpy", label="Загрузите фото полки"),
    outputs=[
        gr.Image(type="numpy", label="Результат анализа (красным выделены пустоты)"),
        gr.Textbox(label="Статус полки")
    ],
    title="🔎 Детекция пустот на полках",
    description="Загрузите изображение полки магазина. Нейросеть U-Net проанализирует заполненность и сообщит, нужно ли выкладывать товар."
)

if __name__ == "__main__":
    demo.launch()
