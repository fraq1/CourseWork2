import streamlit as st
import torch
import numpy as np
import cv2
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 256

st.title(" Проверка заполненности полки")


@st.cache_resource
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
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])


def predict(image):

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    aug = transform(image=image_rgb)
    x = aug["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(x)
        pred = torch.sigmoid(pred)
        mask = (pred > 0.5).float().cpu().numpy()[0, 0]

    return image_rgb, mask


def make_overlay(image, mask):

    h, w = image.shape[:2]

    mask_resized = cv2.resize(mask.astype(np.uint8), (w, h))

    overlay = image.copy()

    overlay[mask_resized == 1] = [255, 0, 0]

    result = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

    return result


uploaded_file = st.file_uploader("Загрузите фото полки", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.subheader("Исходное изображение")
    st.image(image, channels="BGR")

    image_rgb, mask = predict(image)

    result = make_overlay(image_rgb, mask)

    empty_percent = (mask.sum() / mask.size) * 100

    st.subheader("Результат анализа")
    st.image(result)

    st.write(f"Пустоты: {empty_percent:.2f}%")

    if empty_percent > 15:
        st.error("НУЖНО ПОПОЛНЕНИЕ ПОЛКИ")
    else:
        st.success("Полка заполнена")