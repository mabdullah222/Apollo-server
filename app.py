import streamlit as st
import json
import os

files_directory = 'outputs'
json_files = [f for f in os.listdir(files_directory) if f.endswith('.json')]

st.sidebar.title("Lectures")
lecture_names = [file.replace('.json', '') for file in json_files]
selected_lecture = st.sidebar.selectbox("Select a lecture", lecture_names)

selected_file_path = os.path.join(files_directory, f"{selected_lecture}.json")

try:
    with open(selected_file_path, 'r') as ofile:
        doc = json.load(ofile)
        slides = doc['slides']
except FileNotFoundError:
    st.error("Lecture file not found.")
    slides = []

if "slide_index" not in st.session_state:
    st.session_state.slide_index = 0

if st.session_state.slide_index >= len(slides):
    st.session_state.slide_index = 0

if slides:
    current_slide = slides[st.session_state.slide_index]

    for slide in current_slide:
        st.title(slide["title"])
        st.markdown(slide["content"])

        if slide["code"]:
            st.code(slide["code"], language="cpp")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Previous"):
            if st.session_state.slide_index > 0:
                st.session_state.slide_index -= 1
            else:
                st.session_state.slide_index = len(slides) - 1
            st.rerun()

    with col2:
        if st.button("Next"):
            if st.session_state.slide_index < len(slides) - 1:
                st.session_state.slide_index += 1
            else:
                st.session_state.slide_index = 0
            st.rerun()
else:
    st.warning("No slides available for the selected lecture.")
