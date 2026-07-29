import time
import json
import base64
from io import BytesIO

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
import streamlit.components.v1 as components
from gtts import gTTS


CLASS_NAMES = [
    "Pepper Bell - Bacterial Spot",
    "Pepper Bell - Healthy",
    "Potato - Early Blight",
    "Potato - Late Blight",
    "Potato - Healthy",
    "Tomato - Bacterial Spot",
    "Tomato - Early Blight",
    "Tomato - Late Blight",
    "Tomato - Leaf Mold",
    "Tomato - Septoria Leaf Spot",
    "Tomato - Spider Mites",
    "Tomato - Target Spot",
    "Tomato - Yellow Leaf Curl Virus",
    "Tomato - Mosaic Virus",
    "Tomato - Healthy",
]


TRANSLATIONS = {
    "English": {
        "title": "Agri Vision",
        "subtitle": "Check your crop health using a photo.",
        "language": "Language",
        "upload_info": "Take a clear photo of a crop leaf and upload it below.",
        "photo_tips": "Tips for taking a good photo",
        "tip_1": "Take the photo in good daylight.",
        "tip_2": "Make sure the leaf is clearly visible.",
        "tip_3": "Keep the camera steady so the photo is not blurry.",
        "tip_4": "Try to show the affected part of the leaf.",
        "tip_5": "Avoid covering the leaf with your hand.",
        "upload": "Choose a photo of your crop leaf",
        "photo_caption": "Your crop photo",
        "check": "Check My Crop",
        "checking": "Checking your crop...",
        "your_crop": "Your Crop",
        "possible_problem": "Possible Problem",
        "confidence": "AI confidence",
        "what_to_do": "What You Can Do",
        "other_results": "Other Possible Results",
        "healthy_message": "Your plant looks healthy.",
        "pest_message": "A possible pest problem was found.",
        "disease_message": "A possible crop disease was found.",
        "uncertain": "The result is uncertain. Try taking another clear photo in good light.",
        "dark": "This photo is too dark. Please take another photo in better light.",
        "bright": "This photo is too bright. Please avoid strong direct light and try again.",
        "blurry": "This photo looks blurry. Please hold the camera steady and try again.",
        "read_error": "We could not read this photo. Please try another photo.",
        "general_error": "Something went wrong while checking the photo. Please try again with another image.",
        "preliminary": "This is an AI based preliminary check. For important crop treatment decisions, please confirm the problem with a local agricultural expert.",
        "checked": "Checked in",
        "listen": "Listen",
        "stop": "Stop",
        "ready": "Ready",
        "playing": "Playing",
        "finished": "Finished",
        "stopped": "Stopped",
        "audio_error": "Could not play audio",
        "healthy": "Healthy Plant",
        "disease": "Possible Disease",
        "pest": "Possible Pest Problem",
        "crop": {
            "Pepper": "Pepper",
            "Potato": "Potato",
            "Tomato": "Tomato",
        },
        "condition": {
            "Bacterial Spot": "Bacterial Spot",
            "Healthy Plant": "Healthy Plant",
            "Early Blight": "Early Blight",
            "Late Blight": "Late Blight",
            "Leaf Mold": "Leaf Mold",
            "Septoria Leaf Spot": "Septoria Leaf Spot",
            "Spider Mites": "Spider Mites",
            "Target Spot": "Target Spot",
            "Yellow Leaf Curl Virus": "Yellow Leaf Curl Virus",
            "Mosaic Virus": "Mosaic Virus",
        },
    },
    "हिंदी": {
        "title": "एग्री विज़न",
        "subtitle": "फोटो की मदद से अपनी फसल की सेहत जांचें।",
        "language": "भाषा",
        "upload_info": "फसल के पत्ते की एक साफ फोटो लें और नीचे अपलोड करें।",
        "photo_tips": "अच्छी फोटो लेने के लिए सुझाव",
        "tip_1": "फोटो अच्छी रोशनी में लें।",
        "tip_2": "पत्ता साफ और पूरी तरह दिखाई देना चाहिए।",
        "tip_3": "कैमरा स्थिर रखें ताकि फोटो धुंधली न हो।",
        "tip_4": "जहां समस्या दिखाई दे रही है, उस हिस्से को साफ दिखाएं।",
        "tip_5": "फोटो लेते समय पत्ते को हाथ से न ढकें।",
        "upload": "अपनी फसल के पत्ते की फोटो चुनें",
        "photo_caption": "आपकी फसल की फोटो",
        "check": "अपनी फसल की जांच करें",
        "checking": "आपकी फसल की जांच हो रही है...",
        "your_crop": "आपकी फसल",
        "possible_problem": "संभावित समस्या",
        "confidence": "AI का भरोसा",
        "what_to_do": "आप क्या कर सकते हैं",
        "other_results": "अन्य संभावित परिणाम",
        "healthy_message": "आपका पौधा स्वस्थ दिखाई दे रहा है।",
        "pest_message": "पौधे में कीड़ों की समस्या हो सकती है।",
        "disease_message": "पौधे में बीमारी के लक्षण हो सकते हैं।",
        "uncertain": "परिणाम पूरी तरह निश्चित नहीं है। अच्छी रोशनी में पत्ते की एक और साफ फोटो लेने की कोशिश करें।",
        "dark": "यह फोटो बहुत अंधेरी है। कृपया अच्छी रोशनी में दूसरी फोटो लें।",
        "bright": "यह फोटो बहुत ज्यादा चमकीली है। तेज सीधी रोशनी से बचें और दोबारा कोशिश करें।",
        "blurry": "यह फोटो धुंधली दिखाई दे रही है। कैमरा स्थिर रखें और दोबारा कोशिश करें।",
        "read_error": "हम इस फोटो को पढ़ नहीं सके। कृपया दूसरी फोटो चुनें।",
        "general_error": "फोटो की जांच करते समय कुछ समस्या हुई। कृपया दूसरी फोटो के साथ फिर से कोशिश करें।",
        "preliminary": "यह AI द्वारा की गई प्रारंभिक जांच है। फसल के इलाज से जुड़े महत्वपूर्ण फैसले लेने से पहले कृषि विशेषज्ञ से सलाह लें।",
        "checked": "जांच का समय",
        "listen": "सुनें",
        "stop": "रोकें",
        "ready": "तैयार",
        "playing": "आवाज़ चल रही है",
        "finished": "आवाज़ पूरी हो गई",
        "stopped": "आवाज़ रोक दी गई",
        "audio_error": "आवाज़ नहीं चल सकी",
        "healthy": "पौधा स्वस्थ है",
        "disease": "संभावित बीमारी",
        "pest": "संभावित कीड़ों की समस्या",
        "crop": {
            "Pepper": "मिर्च",
            "Potato": "आलू",
            "Tomato": "टमाटर",
        },
        "condition": {
            "Bacterial Spot": "बैक्टीरियल स्पॉट",
            "Healthy Plant": "स्वस्थ पौधा",
            "Early Blight": "अर्ली ब्लाइट",
            "Late Blight": "लेट ब्लाइट",
            "Leaf Mold": "लीफ मोल्ड",
            "Septoria Leaf Spot": "सेप्टोरिया लीफ स्पॉट",
            "Spider Mites": "स्पाइडर माइट्स",
            "Target Spot": "टारगेट स्पॉट",
            "Yellow Leaf Curl Virus": "येलो लीफ कर्ल वायरस",
            "Mosaic Virus": "मोज़ेक वायरस",
        },
    },
}


DISEASE_INFO = {
    "Pepper Bell - Bacterial Spot": {
        "crop": "Pepper",
        "condition": "Bacterial Spot",
        "type": "disease",
        "advice": {
            "English": [
                "Check nearby pepper plants for similar spots.",
                "Watch the affected leaves regularly.",
                "Check if the spots are spreading to other plants.",
                "Ask a local agricultural worker for treatment advice.",
            ],
            "हिंदी": [
                "आसपास के मिर्च के पौधों पर भी ऐसे धब्बे देखें।",
                "प्रभावित पत्तियों पर नियमित रूप से नजर रखें।",
                "देखें कि धब्बे दूसरे पौधों तक तो नहीं फैल रहे हैं।",
                "इलाज की सलाह के लिए स्थानीय कृषि विशेषज्ञ से संपर्क करें।",
            ],
        },
    },
    "Pepper Bell - Healthy": {
        "crop": "Pepper",
        "condition": "Healthy Plant",
        "type": "healthy",
        "advice": {
            "English": [
                "Your plant looks healthy.",
                "Continue checking your plants regularly.",
                "Keep the crop properly watered.",
                "Watch for any new spots, color changes, or damaged leaves.",
            ],
            "हिंदी": [
                "आपका पौधा स्वस्थ दिखाई दे रहा है।",
                "अपने पौधों की नियमित रूप से जांच करते रहें।",
                "फसल को उचित मात्रा में पानी देते रहें।",
                "नए धब्बों, रंग में बदलाव या खराब पत्तियों पर नजर रखें।",
            ],
        },
    },
    "Potato - Early Blight": {
        "crop": "Potato",
        "condition": "Early Blight",
        "type": "disease",
        "advice": {
            "English": [
                "Check nearby potato plants for similar symptoms.",
                "Watch the affected leaves regularly.",
                "Check if the spots are spreading to other plants.",
                "Ask a local agricultural worker for treatment advice.",
            ],
            "हिंदी": [
                "आसपास के आलू के पौधों पर भी ऐसे लक्षण देखें।",
                "प्रभावित पत्तियों पर नियमित रूप से नजर रखें।",
                "देखें कि धब्बे दूसरे पौधों तक तो नहीं फैल रहे हैं।",
                "इलाज की सलाह के लिए स्थानीय कृषि विशेषज्ञ से संपर्क करें।",
            ],
        },
    },
    "Potato - Late Blight": {
        "crop": "Potato",
        "condition": "Late Blight",
        "type": "disease",
        "advice": {
            "English": [
                "Check nearby potato plants for similar symptoms.",
                "Watch the crop closely for new affected leaves.",
                "Check if the disease is spreading quickly.",
                "Ask a local agricultural worker for treatment advice.",
            ],
            "हिंदी": [
                "आसपास के आलू के पौधों पर भी ऐसे लक्षण देखें।",
                "नई प्रभावित पत्तियों के लिए फसल पर ध्यान से नजर रखें।",
                "देखें कि बीमारी तेजी से दूसरे पौधों तक तो नहीं फैल रही है।",
                "इलाज की सलाह के लिए स्थानीय कृषि विशेषज्ञ से संपर्क करें।",
            ],
        },
    },
    "Potato - Healthy": {
        "crop": "Potato",
        "condition": "Healthy Plant",
        "type": "healthy",
        "advice": {
            "English": [
                "Your plant looks healthy.",
                "Continue checking your plants regularly.",
                "Keep the crop properly watered.",
                "Watch for any new spots or color changes.",
            ],
            "हिंदी": [
                "आपका पौधा स्वस्थ दिखाई दे रहा है।",
                "अपने पौधों की नियमित रूप से जांच करते रहें।",
                "फसल को उचित मात्रा में पानी देते रहें।",
                "नए धब्बों या रंग में बदलाव पर नजर रखें।",
            ],
        },
    },
    "Tomato - Bacterial Spot": {
        "crop": "Tomato",
        "condition": "Bacterial Spot",
        "type": "disease",
        "advice": {
            "English": [
                "Check nearby tomato plants for similar spots.",
                "Watch the affected leaves regularly.",
                "Check if the spots are spreading.",
                "Ask a local agricultural worker for treatment advice.",
            ],
            "हिंदी": [
                "आसपास के टमाटर के पौधों पर भी ऐसे धब्बे देखें।",
                "प्रभावित पत्तियों पर नियमित रूप से नजर रखें।",
                "देखें कि धब्बे फैल तो नहीं रहे हैं।",
                "इलाज की सलाह के लिए स्थानीय कृषि विशेषज्ञ से संपर्क करें।",
            ],
        },
    },
    "Tomato - Early Blight": {
        "crop": "Tomato",
        "condition": "Early Blight",
        "type": "disease",
        "advice": {
            "English": [
                "Check nearby tomato plants for similar symptoms.",
                "Watch the affected leaves regularly.",
                "Check other plants for similar spots.",
                "Ask a local agricultural worker for treatment advice.",
            ],
            "हिंदी": [
                "आसपास के टमाटर के पौधों पर भी ऐसे लक्षण देखें।",
                "प्रभावित पत्तियों पर नियमित रूप से नजर रखें।",
                "दूसरे पौधों पर भी ऐसे धब्बे देखें।",
                "इलाज की सलाह के लिए स्थानीय कृषि विशेषज्ञ से संपर्क करें।",
            ],
        },
    },
    "Tomato - Late Blight": {
        "crop": "Tomato",
        "condition": "Late Blight",
        "type": "disease",
        "advice": {
            "English": [
                "Check nearby tomato plants for similar symptoms.",
                "Watch the crop closely for new affected leaves.",
                "Check if the disease is spreading to other plants.",
                "Ask a local agricultural worker for treatment advice.",
            ],
            "हिंदी": [
                "आसपास के टमाटर के पौधों पर भी ऐसे लक्षण देखें।",
                "नई प्रभावित पत्तियों के लिए फसल पर ध्यान से नजर रखें।",
                "देखें कि बीमारी दूसरे पौधों तक तो नहीं फैल रही है।",
                "इलाज की सलाह के लिए स्थानीय कृषि विशेषज्ञ से संपर्क करें।",
            ],
        },
    },
    "Tomato - Leaf Mold": {
        "crop": "Tomato",
        "condition": "Leaf Mold",
        "type": "disease",
        "advice": {
            "English": [
                "Check nearby tomato plants for similar symptoms.",
                "Watch the affected leaves regularly.",
                "Check if more leaves are becoming affected.",
                "Ask a local agricultural worker for treatment advice.",
            ],
            "हिंदी": [
                "आसपास के टमाटर के पौधों पर भी ऐसे लक्षण देखें।",
                "प्रभावित पत्तियों पर नियमित रूप से नजर रखें।",
                "देखें कि और पत्तियां भी प्रभावित तो नहीं हो रही हैं।",
                "इलाज की सलाह के लिए स्थानीय कृषि विशेषज्ञ से संपर्क करें।",
            ],
        },
    },
    "Tomato - Septoria Leaf Spot": {
        "crop": "Tomato",
        "condition": "Septoria Leaf Spot",
        "type": "disease",
        "advice": {
            "English": [
                "Check nearby tomato plants for similar spots.",
                "Watch the lower leaves carefully.",
                "Check nearby plants for similar symptoms.",
                "Ask a local agricultural worker for treatment advice.",
            ],
            "हिंदी": [
                "आसपास के टमाटर के पौधों पर भी ऐसे धब्बे देखें।",
                "नीचे की पत्तियों को ध्यान से देखें।",
                "आसपास के पौधों पर भी ऐसे लक्षण देखें।",
                "इलाज की सलाह के लिए स्थानीय कृषि विशेषज्ञ से संपर्क करें।",
            ],
        },
    },
    "Tomato - Spider Mites": {
        "crop": "Tomato",
        "condition": "Spider Mites",
        "type": "pest",
        "advice": {
            "English": [
                "Look carefully under the leaves for small insects.",
                "Check nearby plants for similar signs.",
                "Watch the plants for increasing pest activity.",
                "Ask a local agricultural worker for treatment advice.",
            ],
            "हिंदी": [
                "छोटे कीड़ों के लिए पत्तियों के नीचे ध्यान से देखें।",
                "आसपास के पौधों पर भी ऐसे लक्षण देखें।",
                "पौधों पर कीड़ों की समस्या बढ़ रही है या नहीं, इस पर नजर रखें।",
                "इलाज की सलाह के लिए स्थानीय कृषि विशेषज्ञ से संपर्क करें।",
            ],
        },
    },
    "Tomato - Target Spot": {
        "crop": "Tomato",
        "condition": "Target Spot",
        "type": "disease",
        "advice": {
            "English": [
                "Check nearby tomato plants for similar spots.",
                "Watch the affected leaves regularly.",
                "Check if the spots are spreading.",
                "Ask a local agricultural worker for treatment advice.",
            ],
            "हिंदी": [
                "आसपास के टमाटर के पौधों पर भी ऐसे धब्बे देखें।",
                "प्रभावित पत्तियों पर नियमित रूप से नजर रखें।",
                "देखें कि धब्बे फैल तो नहीं रहे हैं।",
                "इलाज की सलाह के लिए स्थानीय कृषि विशेषज्ञ से संपर्क करें।",
            ],
        },
    },
    "Tomato - Yellow Leaf Curl Virus": {
        "crop": "Tomato",
        "condition": "Yellow Leaf Curl Virus",
        "type": "disease",
        "advice": {
            "English": [
                "Check nearby tomato plants for similar symptoms.",
                "Look for other plants with curled or yellow leaves.",
                "Watch the crop for new affected plants.",
                "Ask a local agricultural worker for advice.",
            ],
            "हिंदी": [
                "आसपास के टमाटर के पौधों पर भी ऐसे लक्षण देखें।",
                "ऐसे पौधों को देखें जिनकी पत्तियां मुड़ी हुई या पीली हैं।",
                "नई प्रभावित पौधों के लिए फसल पर नजर रखें।",
                "सलाह के लिए स्थानीय कृषि विशेषज्ञ से संपर्क करें।",
            ],
        },
    },
    "Tomato - Mosaic Virus": {
        "crop": "Tomato",
        "condition": "Mosaic Virus",
        "type": "disease",
        "advice": {
            "English": [
                "Check nearby tomato plants for similar symptoms.",
                "Watch for unusual patterns or color changes on leaves.",
                "Keep checking the crop regularly.",
                "Ask a local agricultural worker for advice.",
            ],
            "हिंदी": [
                "आसपास के टमाटर के पौधों पर भी ऐसे लक्षण देखें।",
                "पत्तियों पर असामान्य पैटर्न या रंग में बदलाव देखें।",
                "फसल की नियमित रूप से जांच करते रहें।",
                "सलाह के लिए स्थानीय कृषि विशेषज्ञ से संपर्क करें।",
            ],
        },
    },
    "Tomato - Healthy": {
        "crop": "Tomato",
        "condition": "Healthy Plant",
        "type": "healthy",
        "advice": {
            "English": [
                "Your plant looks healthy.",
                "Continue checking your plants regularly.",
                "Keep the crop properly watered.",
                "Watch for any new spots, color changes, or damaged leaves.",
            ],
            "हिंदी": [
                "आपका पौधा स्वस्थ दिखाई दे रहा है।",
                "अपने पौधों की नियमित रूप से जांच करते रहें।",
                "फसल को उचित मात्रा में पानी देते रहें।",
                "नए धब्बों, रंग में बदलाव या खराब पत्तियों पर नजर रखें।",
            ],
        },
    },
}


st.set_page_config(
    page_title="Agri Vision",
    layout="centered",
)

st.markdown(
    """
    <style>
    .main {
        max-width: 760px;
        margin: auto;
    }

    .title {
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 4px;
        text-align: left;
    }

    .subtitle {
        font-size: 18px;
        margin-bottom: 28px;
        opacity: 0.8;
        text-align: left;
    }

    .section-label {
        font-size: 15px;
        font-weight: 600;
        margin-top: 24px;
        margin-bottom: 4px;
        text-align: left;
    }

    .result-value {
        font-size: 26px;
        font-weight: 700;
        margin-bottom: 18px;
        text-align: left;
    }

    .condition-value {
        font-size: 30px;
        font-weight: 700;
        margin-bottom: 8px;
        text-align: left;
    }

    .confidence {
        font-size: 15px;
        opacity: 0.75;
        margin-bottom: 20px;
        text-align: left;
    }

    .tts-card {
        width: 100%;
        box-sizing: border-box;
        border: 1px solid #d6d9dd;
        border-radius: 12px;
        padding: 14px;
        margin-top: 12px;
        margin-bottom: 12px;
        background: #ffffff;
    }

    .tts-title {
        font-size: 14px;
        font-weight: 600;
        color: #333333;
        margin-bottom: 10px;
        text-align: left;
    }

    .tts-row {
        display: flex;
        gap: 10px;
        width: 100%;
    }

    .tts-btn {
        flex: 1;
        min-height: 44px;
        padding: 10px 16px;
        border-radius: 9px;
        border: 1px solid #b8bdc5;
        font-size: 15px;
        font-weight: 600;
        cursor: pointer;
        transition: 0.15s;
    }

    .tts-listen {
        background: #166534;
        color: white;
        border-color: #166534;
    }

    .tts-listen:hover {
        background: #14532d;
    }

    .tts-stop {
        background: #ffffff;
        color: #222222;
    }

    .tts-stop:hover {
        background: #f2f2f2;
    }

    .tts-status {
        margin-top: 10px;
        font-size: 13px;
        color: #666666;
        text-align: left;
    }

    @media (max-width: 500px) {
        .tts-row {
            flex-direction: column;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("crop_disease_model.h5")


@st.cache_data(show_spinner=False)
def generate_tts_audio(text_to_speak, language):
    lang = "hi" if language == "हिंदी" else "en"
    fp = BytesIO()
    gTTS(text=text_to_speak, lang=lang, slow=False).write_to_fp(fp)
    return base64.b64encode(fp.getvalue()).decode("utf-8")


def prepare_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.getvalue()), dtype=np.uint8)

    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        return None, None

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    brightness = np.mean(gray_image)
    blur_score = cv2.Laplacian(gray_image, cv2.CV_64F).var()

    processed_image = cv2.resize(rgb_image, (224, 224))
    processed_image = processed_image.astype(np.float32) / 255.0
    processed_image = np.expand_dims(processed_image, axis=0)

    image_quality = {
        "brightness": brightness,
        "blur_score": blur_score,
    }

    return processed_image, image_quality


def check_image_quality(image_quality):
    brightness = image_quality["brightness"]
    blur_score = image_quality["blur_score"]

    if brightness < 35:
        return "dark"

    if brightness > 245:
        return "bright"

    if blur_score < 40:
        return "blurry"

    return "good"


def tts_panel(text_to_speak, language):
    text = TRANSLATIONS[language]
    listen_label = text["listen"]
    stop_label = text["stop"]

    audio_b64 = generate_tts_audio(text_to_speak, language)

    components.html(
        f"""
        <style>
        .tts-row {{
            display: flex;
            gap: 10px;
            margin: 8px 0 0 0;
            width: 100%;
        }}

        .tts-btn {{
            flex: 1;
            height: 42px;
            border-radius: 10px;
            border: 1px solid #d1d5db;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            background: #ffffff;
            color: #111827;
            transition: background 0.15s ease, border-color 0.15s ease, transform 0.05s ease;
        }}

        .tts-btn:active {{
            transform: scale(0.98);
        }}

        .tts-btn.listen {{
            background: #111827;
            color: #ffffff;
            border-color: #111827;
        }}

        .tts-btn.listen:hover {{
            background: #1f2937;
        }}

        .tts-btn.stop:hover {{
            background: #f3f4f6;
        }}

        @media (max-width: 480px) {{
            .tts-row {{
                flex-direction: column;
            }}

            .tts-btn {{
                width: 100%;
            }}
        }}
        </style>

        <div class="tts-row">
            <button class="tts-btn listen" onclick="playAudio()">{listen_label}</button>
            <button class="tts-btn stop" onclick="stopAudio()">{stop_label}</button>
        </div>

        <audio id="player" preload="auto" style="display:none;">
            <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
        </audio>

        <script>
        const player = document.getElementById("player");

        function playAudio() {{
            player.pause();
            player.currentTime = 0;
            player.play();
        }}

        function stopAudio() {{
            player.pause();
            player.currentTime = 0;
        }}
        </script>
        """,
        height=60,
    )

def translate_prediction_label(label, language):
    text = TRANSLATIONS[language]
    parts = label.split(" - ", 1)

    if len(parts) != 2:
        return label

    crop = parts[0]
    condition = parts[1]

    translated_crop = text["crop"].get(crop, crop)
    translated_condition = text["condition"].get(condition, condition)

    return translated_crop + " - " + translated_condition


def show_result(predicted_class, confidence, processing_time, prediction, language):
    info = DISEASE_INFO[predicted_class]
    text = TRANSLATIONS[language]

    crop_name = text["crop"].get(info["crop"], info["crop"])
    condition_name = text["condition"].get(info["condition"], info["condition"])
    advice = info["advice"][language]

    if info["type"] == "healthy":
        status_message = text["healthy_message"]
    elif info["type"] == "pest":
        status_message = text["pest_message"]
    else:
        status_message = text["disease_message"]

    speech_text = (
        f"{text['your_crop']}: {crop_name}. "
        f"{text['possible_problem']}: {condition_name}. "
        f"{text['what_to_do']}. "
        + " ".join(advice)
    )

    st.divider()

    st.markdown(f'<div class="section-label">{text["your_crop"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="result-value">{crop_name}</div>', unsafe_allow_html=True)

    st.markdown(f'<div class="section-label">{text["possible_problem"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="condition-value">{condition_name}</div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="confidence">{text["confidence"]}: {confidence:.2f}%</div>',
        unsafe_allow_html=True,
    )

    if info["type"] == "healthy":
        st.success(status_message)
    elif info["type"] == "pest":
        st.warning(status_message)
    else:
        st.error(status_message)

    if confidence < 60:
        st.warning(text["uncertain"])

    st.markdown(f'<div class="section-label">{text["what_to_do"]}</div>', unsafe_allow_html=True)

    for number, advice_item in enumerate(advice, start=1):
        st.write(f"{number}. {advice_item}")

    tts_panel(speech_text, language)

    st.divider()

    st.markdown(f'<div class="section-label">{text["other_results"]}</div>', unsafe_allow_html=True)

    top_indices = np.argsort(prediction)[-3:][::-1]

    for index in top_indices:
        label = CLASS_NAMES[index]
        score = float(prediction[index] * 100)
        display_label = translate_prediction_label(label, language)

        st.write(display_label)
        st.progress(min(int(score), 100))
        st.caption(f"{score:.2f}%")

    st.info(text["preliminary"])
    st.caption(f'{text["checked"]}: {processing_time:.2f} seconds.')


language = st.selectbox("Language / भाषा", ["English", "हिंदी"])
text = TRANSLATIONS[language]

st.markdown(f'<div class="title">{text["title"]}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="subtitle">{text["subtitle"]}</div>', unsafe_allow_html=True)

st.info(text["upload_info"])

with st.expander(text["photo_tips"]):
    st.write(text["tip_1"])
    st.write(text["tip_2"])
    st.write(text["tip_3"])
    st.write(text["tip_4"])
    st.write(text["tip_5"])

uploaded_file = st.file_uploader(
    text["upload"],
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    st.image(uploaded_file, caption=text["photo_caption"], use_container_width=True)

    analyze = st.button(text["check"], type="primary", use_container_width=True)

    if analyze:
        try:
            with st.spinner(text["checking"]):
                start_time = time.time()
                model = load_model()
                input_image, image_quality = prepare_image(uploaded_file)

                if input_image is None:
                    st.error(text["read_error"])
                    st.stop()

                quality = check_image_quality(image_quality)

                if quality == "dark":
                    st.warning(text["dark"])
                    st.stop()

                if quality == "bright":
                    st.warning(text["bright"])
                    st.stop()

                if quality == "blurry":
                    st.warning(text["blurry"])
                    st.stop()

                prediction = model.predict(input_image, verbose=0)[0]
                predicted_index = int(np.argmax(prediction))
                predicted_class = CLASS_NAMES[predicted_index]
                confidence = float(np.max(prediction) * 100)
                processing_time = time.time() - start_time

            show_result(predicted_class, confidence, processing_time, prediction, language)

        except Exception:
            st.error(text["general_error"])