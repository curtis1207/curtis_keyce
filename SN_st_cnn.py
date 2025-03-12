import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import os
import io
import numpy as np

class KClassifier(nn.Module):
    def __init__(self, dim_output=4):
        super().__init__()

        self.conv_relu_stack = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),  # (250 - 3 + 2*0)/1 + 1 = 248
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (248 - 2) / 2 + 1 = 128
        )

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features=32*124*124, out_features=500),
            nn.ReLU(),
            nn.Linear(in_features=500, out_features=dim_output)
        )

    def forward(self, x):
        x = self.conv_relu_stack(x)
        x = torch.flatten(x, 1)
        logits = self.linear_relu_stack(x)
        return logits

# Charger le modèle avec st.cache_resource sauvegarder
@st.cache_resource
def load_model():
    model = KClassifier()
    model.load_state_dict(torch.load("model_weights.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Vérifier que le modèle a bien été chargé
if model is None:
    st.error("Le modèle n'a pas pu être chargé. Veuillez vérifier le fichier du modèle.")
else:
    st.success("Modèle chargé avec succès.")

# Dictionnaire des noms de classes (ajustez-le selon vos classes réelles)
class_names = {
    0: "Cyst",
    1: "Normal",
    2: "Stone",
    3: "Tumor"
}

# Interface Streamlit pour télécharger l'image
st.sidebar.title("🩺 Classification des Images")
st.sidebar.markdown("Téléchargez une image pour classification.")

uploaded_file = st.sidebar.file_uploader("Uploader une image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Charger et afficher l'image
        image_data = uploaded_file.read()
        img = Image.open(io.BytesIO(image_data)).convert('RGB')
        st.image(img, caption="Image téléchargée", use_container_width=True)

        # Prétraiter l'image
        target_size = (250, 250)  # Utiliser une taille compatible avec le modèle
        img = img.resize(target_size)

        # Transformation : Convertir en Tensor et Normaliser
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalisation pour 3 canaux
        ])
        
        img_tensor = transform(img).unsqueeze(0)  # Ajouter la dimension du lot : [1, C, H, W]
        
        # Vérifier que le modèle est correctement chargé avant de faire une prédiction
        if model:
            # Faire la prédiction des données
            with torch.no_grad():  # Pas besoin de calculer les gradients
                preds = model(img_tensor)  # Prédiction avec le modèle

            # Récupérer les résultats
            predicted_class_idx = np.argmax(preds.detach().numpy())  # Index de la classe prédite
            confidence = np.max(preds.detach().numpy()) * 100  # Confiance

            # Afficher le résultat
            st.subheader("Résultats de la classification")
            st.write(f"**Classe prédite :** {class_names[predicted_class_idx]}")
            st.write(f"**Confiance :** {confidence:.2f}%")
        else:
            st.error("Le modèle est indisponible pour faire une prédiction.")

    except Exception as e:
        st.error(f"Une erreur est survenue pendant le traitement de l'image: {e}")
