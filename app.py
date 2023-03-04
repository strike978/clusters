import io
import pandas as pd
import streamlit as st
from sklearn.cluster import AgglomerativeClustering, KMeans

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

# Add a row with clustering method and number of clusters
col1, col2 = st.columns(2)
with col1:
    clustering_method = st.selectbox("Clustering method:", ["Ward", "K-means"])
with col2:
    n_clusters = st.number_input(
        "Number of clusters:", min_value=2, max_value=64, value=3)

# Add a button to perform clustering and display results
if st.button("Cluster populations"):
    # Load data from CSV data
    data = pd.read_csv(io.StringIO(csv_data), header=None)
    data.set_index(data.columns[0], inplace=True)

    # Perform clustering using the selected method
    if clustering_method == "Ward":
        cluster = AgglomerativeClustering(
            n_clusters=n_clusters, linkage="ward")
    else:
        cluster = KMeans(n_clusters=n_clusters)
    cluster.fit(data)

    # Create a dictionary of population lists, grouped by cluster
    pop_clusters = {i+1: [] for i in range(cluster.n_clusters)}
    for pop, c in zip(data.index, cluster.labels_):
        pop_clusters[c+1].append(pop)

    # Print the populations in each cluster
    output = f"Clustering method: {clustering_method}\n\n"
    for c, pops in pop_clusters.items():
        output += f"Cluster {c}: {', '.join(pops)}\n"
        if c < len(pop_clusters):
            output += "\n"
    st.code(output)

    # Add a download button for the output
    file_name = f"clustering_output_{clustering_method}_{n_clusters}.txt"
    file_bytes = output.encode("utf-8")
    st.download_button(label="Download output",
                       data=file_bytes, file_name=file_name)
