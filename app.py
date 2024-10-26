import streamlit as st

st.set_page_config(
    page_title="Test",
    page_icon="🌚",
)

pg = st.navigation([st.Page("page_1.py"), st.Page("page_2.py")])
pg.run()
