import streamlit as st

st.title("Test Application")
st.write("If you can see this, Streamlit is working correctly!")

st.header("Basic Input Test")
name = st.text_input("Enter your name")
if name:
    st.write(f"Hello, {name}!")

st.header("Simple Chart Test")
import numpy as np
import pandas as pd
import plotly.express as px

# Generate some random data
data = np.random.randn(100)
df = pd.DataFrame({"values": data})

# Create a simple chart
fig = px.histogram(df, x="values", title="Simple Histogram")
st.plotly_chart(fig)

st.write("If you can see everything above, your basic dependencies are working!") 