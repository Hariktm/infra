import streamlit as st


pages = {

    "Wave Infra":[
        st.Page("appp.py",title="Excel Analyzer"),
        st.Page("Checkreport.py",title="Checklist"),
        st.Page("ncr.py",title="NCR"),
        # st.Page("reports.py",title="Reports",icon=":material/report:")
    ]
}


st.navigation(pages).run()

