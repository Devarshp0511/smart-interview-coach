import streamlit as st
import os
from utils import extract_audio, transcribe_audio, analyze_video_eye_contact, analyze_audio_tone, generate_feedback

if not os.path.exists("temp_data"):
    os.makedirs("temp_data")

# Configure Page
st.set_page_config(page_title="AI Interview Coach", layout="wide")

st.title("🤖 AI Interview Coach")
st.write("Upload your interview answer video to get instant feedback on your content, visual focus, and voice tone.")

# File Uploader
uploaded_file = st.file_uploader("Upload MP4 Video", type=["mp4", "mov"])

if uploaded_file is not None:
    # Save file temporarily
    video_path = f"temp_data/{uploaded_file.name}"
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.video(video_path)
    
    if st.button("Analyze My Answer"):
        with st.spinner("Analyzing Video, Audio, and Tone..."):
            # 1. Extract Audio
            audio_path = extract_audio(video_path)
            
            # 2. Transcribe Audio
            transcript = transcribe_audio(audio_path)
            
            # 3. Analyze Audio Tone (NEW)
            tone_analysis = analyze_audio_tone(audio_path)
            
            # 4. Analyze Vision (Returns score + image path)
            eye_contact, debug_img_path = analyze_video_eye_contact(video_path)
            
            # 5. Get AI Feedback (LLM)
            feedback = generate_feedback(transcript, eye_contact)
            
            # --- Display Results ---
            st.success("Analysis Complete!")
            
            # Three Columns for Metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Eye Contact Score", f"{eye_contact:.1f}%")
            
            with col2:
                # NEW: Color-coded Tone Metric
                st.metric("Voice Tone", tone_analysis)
                if "Monotone" in tone_analysis:
                    st.caption("⚠️ Try to vary your pitch more.")
                else:
                    st.caption("✅ Good vocal variety.")
            
            with col3:
                # Show the "Computer Vision" view
                if debug_img_path and os.path.exists(debug_img_path):
                    st.image(debug_img_path, caption="AI Vision Tracking", width=200)
            
            st.divider()
            
            # Transcript Section
            st.subheader("📝 Transcript")
            st.info(transcript)
            
            # AI Feedback Section
            st.subheader("💡 AI Coach Feedback")
            st.markdown(feedback)
            
            # Cleanup (Optional)
            if os.path.exists(audio_path): 
                os.remove(audio_path)