import io
import pandas as pd
import streamlit as st
from sklearn.cluster import AgglomerativeClustering

# Define the app interface
st.set_page_config(layout="wide", page_title="G25 Clustering App")
st.title("G25 Clustering App")

# Load default data from "africa.csv"
with open("africa.csv", "r") as f:
    default_csv_data = f.read()

# Add a file uploader to change the input data
uploaded_file = st.file_uploader(
    "Upload CSV or TXT file:", type=["csv", "txt"])

if uploaded_file is not None:
    # Load data from uploaded file
    csv_data = uploaded_file.read().decode("utf-8")
else:
    # Use default data from "africa.csv"
    csv_data = default_csv_data

# Add a text input box for the user to edit the CSV data
csv_data = st.text_area("Edit CSV data:", value=csv_data, height=300)

# Add a number input to select the number of clusters
n_clusters = st.number_input(
    "Number of clusters:", min_value=2, max_value=64, value=3)

# Add a button to perform clustering and display results
if st.button("Cluster populations"):
    # Load data from CSV data
    data = pd.read_csv(io.StringIO(csv_data), index_col=0)

    # Perform hierarchical clustering using Ward's method
    cluster = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    cluster.fit(data)

    # Create a dictionary of population lists, grouped by cluster
    pop_clusters = {i+1: [] for i in range(cluster.n_clusters)}
    for pop, c in zip(data.index, cluster.labels_):
        pop_clusters[c+1].append(pop)

    # Print the populations in each cluster
    output = ""
    for c, pops in pop_clusters.items():
        output += f"Cluster {c}: {', '.join(pops)}\n"
    st.code(output)
