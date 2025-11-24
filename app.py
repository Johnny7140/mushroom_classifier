import streamlit as st
import joblib
import numpy as np
from tensorflow import keras
from PIL import Image

# --- Load preprocessing + model ---
encoder = joblib.load("encoder.pkl")
pca = joblib.load("pca.pkl")
model = keras.models.load_model("pca_mushroom_model.h5")

st.title("🍄 Mushroom Classifier")
st.write("Educational tool for kids: edible vs poisonous mushrooms")


# --- Image upload ---
uploaded_image = st.file_uploader("Upload a mushroom photo", type=["jpg","jpeg","png"])
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Mushroom", use_column_width=True)
    st.info("Note: Image is displayed for context. Classification still uses dropdown features.")



# --- Dropdowns with dataset codes and friendly labels ---
cap_shape = st.selectbox("Cap Shape", ["b","c","x","f","k","s"], format_func=lambda x: {"b":"bell","c":"conical","x":"convex","f":"flat","k":"knobbed","s":"sunken"}[x])
cap_surface = st.selectbox("Cap Surface", ["f","g","y","s"], format_func=lambda x: {"f":"fibrous","g":"grooves","y":"scaly","s":"smooth"}[x])
cap_color = st.selectbox("Cap Color", ["n","b","c","g","r","p","u","e","w","y"], format_func=lambda x: {"n":"brown","b":"buff","c":"cinnamon","g":"gray","r":"green","p":"pink","u":"purple","e":"red","w":"white","y":"yellow"}[x])
bruises = st.selectbox("Bruises?", ["t","f"], format_func=lambda x: {"t":"bruises","f":"no"}[x])
odor = st.selectbox("Odor", ["a","l","c","y","f","m","n","p","s"], format_func=lambda x: {"a":"almond","l":"anise","c":"creosote","y":"fishy","f":"foul","m":"musty","n":"none","p":"pungent","s":"spicy"}[x])
gill_attachment = st.selectbox("Gill Attachment", ["a","d","f","n"], format_func=lambda x: {"a":"attached","d":"descending","f":"free","n":"notched"}[x])
gill_spacing = st.selectbox("Gill Spacing", ["c","w","d"], format_func=lambda x: {"c":"close","w":"crowded","d":"distant"}[x])
gill_size = st.selectbox("Gill Size", ["b","n"], format_func=lambda x: {"b":"broad","n":"narrow"}[x])
gill_color = st.selectbox("Gill Color", ["k","n","b","h","g","r","o","p","u","e","w","y"], format_func=lambda x: {"k":"black","n":"brown","b":"buff","h":"chocolate","g":"gray","r":"green","o":"orange","p":"pink","u":"purple","e":"red","w":"white","y":"yellow"}[x])
stalk_shape = st.selectbox("Stalk Shape", ["e","t"], format_func=lambda x: {"e":"enlarging","t":"tapering"}[x])
stalk_root = st.selectbox("Stalk Root", ["b","c","u","e","z","r","?"], format_func=lambda x: {"b":"bulbous","c":"club","u":"cup","e":"equal","z":"rhizomorphs","r":"rooted","?":"missing"}[x])
stalk_surface_above_ring = st.selectbox("Stalk Surface Above Ring", ["f","y","k","s"], format_func=lambda x: {"f":"fibrous","y":"scaly","k":"silky","s":"smooth"}[x])
stalk_surface_below_ring = st.selectbox("Stalk Surface Below Ring", ["f","y","k","s"], format_func=lambda x: {"f":"fibrous","y":"scaly","k":"silky","s":"smooth"}[x])
stalk_color_above_ring = st.selectbox("Stalk Color Above Ring", ["n","b","c","g","o","p","e","w","y"], format_func=lambda x: {"n":"brown","b":"buff","c":"cinnamon","g":"gray","o":"orange","p":"pink","e":"red","w":"white","y":"yellow"}[x])
stalk_color_below_ring = st.selectbox("Stalk Color Below Ring", ["n","b","c","g","o","p","e","w","y"], format_func=lambda x: {"n":"brown","b":"buff","c":"cinnamon","g":"gray","o":"orange","p":"pink","e":"red","w":"white","y":"yellow"}[x])
veil_type = st.selectbox("Veil Type", ["p","u"], format_func=lambda x: {"p":"partial","u":"universal"}[x])
veil_color = st.selectbox("Veil Color", ["n","o","w","y"], format_func=lambda x: {"n":"brown","o":"orange","w":"white","y":"yellow"}[x])
ring_number = st.selectbox("Ring Number", ["n","o","t"], format_func=lambda x: {"n":"none","o":"one","t":"two"}[x])
ring_type = st.selectbox("Ring Type", ["c","e","f","l","n","p","s","z"], format_func=lambda x: {"c":"cobwebby","e":"evanescent","f":"flaring","l":"large","n":"none","p":"pendant","s":"sheathing","z":"zone"}[x])
spore_print_color = st.selectbox("Spore Print Color", ["k","n","b","h","r","o","u","w","y"], format_func=lambda x: {"k":"black","n":"brown","b":"buff","h":"chocolate","r":"green","o":"orange","u":"purple","w":"white","y":"yellow"}[x])
population = st.selectbox("Population", ["a","c","n","s","v","y"], format_func=lambda x: {"a":"abundant","c":"clustered","n":"numerous","s":"scattered","v":"several","y":"solitary"}[x])
habitat = st.selectbox("Habitat", ["g","l","m","p","u","w","d"], format_func=lambda x: {"g":"grasses","l":"leaves","m":"meadows","p":"paths","u":"urban","w":"waste","d":"woods"}[x])

# --- Prediction button ---
if st.button("Predict"):
    raw_features = [[
        cap_shape, cap_surface, cap_color, bruises, odor,
        gill_attachment, gill_spacing, gill_size, gill_color,
        stalk_shape, stalk_root, stalk_surface_above_ring, stalk_surface_below_ring,
        stalk_color_above_ring, stalk_color_below_ring,
        veil_type, veil_color, ring_number, ring_type,
        spore_print_color, population, habitat
    ]]

    st.session_state["last_input"] = raw_features[0]

    encoded = encoder.transform(raw_features)
    if hasattr(encoded, "toarray"):
        encoded = encoded.toarray()

    features = pca.transform(encoded)
    prediction = model.predict(features)
    prob = prediction[0][0]

    st.write("### Prediction Confidence")
    st.progress(int(prob * 100))

    if prob > 0.5:
        st.success(f"✅ Edible Mushroom (confidence {prob:.2f})")
        st.metric(label="Poisonous Confidence", value=f"{(1-prob)*100:.1f}%")
    else:
        st.error(f"⚠️ Poisonous Mushroom (confidence {(1-prob):.2f})")
        st.metric(label="Edible Confidence", value=f"{prob*100:.1f}%")

# --- Why button ---
if st.button("Why?") and "last_input" in st.session_state:
    input_row = st.session_state["last_input"]

    odor_explain = {
        "a": "almond", "l": "anise", "c": "creosote", "y": "fishy",
        "f": "foul", "m": "musty", "n": "none", "p": "pungent", "s": "spicy"
    }
    spore_explain = {
        "k": "black", "n": "brown", "b": "buff", "h": "chocolate",
        "r": "green", "o": "orange", "u": "purple", "w": "white", "y": "yellow"
    }

    odor = odor_explain.get(input_row[4], "unknown")
    spore = spore_explain.get(input_row[20], "unknown")

    st.write("🔍 The model focused on:")
    st.markdown(f"- **Odor**: `{odor}` — a strong signal for edibility")
    st.markdown(f"- **Spore Print Color**: `{spore}` — often linked to mushroom safety")
    st.info("These features had the biggest influence on the prediction.")
