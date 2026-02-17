import streamlit as st
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification

# Load model from local folder
MODEL_PATH = "bert_spam_model"

tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)

model.eval()

st.title("📩 SMS Spam Detection using BERT")

user_input = st.text_area("Enter SMS message:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        inputs = tokenizer(
            user_input,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1)

        if prediction.item() == 1:
            st.error("🚨 Spam Message")
        else:
            st.success("✅ Not Spam (Ham)")
