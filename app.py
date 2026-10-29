import streamlit as st
import joblib

st.set_page_config(page_title="Spam Detector", page_icon="📩")

model = joblib.load("spam_model.joblib")

st.title("📩 Spam Message Detector")
st.write("Enter a message below to check whether it might be spam.")

message = st.text_area("Enter your message:", key="sms_input")

if st.button("Check Message"):
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        prediction = model.predict([message])[0]

        if prediction == "spam":
            st.error("🚨 This message looks like SPAM!")
        else:
            st.success("✅ This message looks like NOT SPAM!")
