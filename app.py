import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
import streamlit as st
import io

# Setting the layout of the page to wide and the title of the page to G25 Clustering.
st.set_page_config(layout="wide", page_title="G25 Clustering")
st.title("G25 Clustering")

# Creating a file uploader to upload data as CSV or text
uploaded_file = st.file_uploader(
    "Upload a CSV or text file", type=["csv", "txt"])

# Reading the data from the file africa.csv and displaying it in the text area.
default_data = open("africa.csv", "r").read()
if uploaded_file is not None:
    default_data = uploaded_file.getvalue().decode('utf-8')

data_input = st.text_area("Enter data in CSV format", value=default_data)

# Reading the data from the file africa.csv and displaying it in the text area.
if data_input:
    data = pd.read_csv(io.StringIO(data_input), header=None).iloc[:, 1:]
    populations = pd.read_csv(io.StringIO(
        data_input), header=None, usecols=[0])[0]

    # This is the code that is used to cluster the data.
    num_clusters = st.number_input(
        "Number of clusters", min_value=2, max_value=len(data), value=2)
    clustering_method = "ward"

    linkage_matrix = linkage(data, method='ward')
    cluster_assignments = fcluster(
        linkage_matrix, num_clusters, criterion='maxclust')

    # Concatenating the populations, cluster assignments and data.
    results = pd.concat(
        [populations, pd.Series(cluster_assignments), data], axis=1)
    results.columns = ['Population', 'Cluster'] + \
        [f"Feature {i}" for i in range(len(data.columns))]
    results_sorted = results.sort_values('Cluster')

    output_str = ""
    for i in range(len(results_sorted)):
        population_str = ':'.join(
            results_sorted.iloc[i]['Population'].split(':')[:4])
        country_str = results_sorted.iloc[i]['Population'].split(':')[-1]
        feature_str = ','.join([str(val)
                               for val in results_sorted.iloc[i][2:]])
        output_str += f"{results_sorted.iloc[i]['Cluster']}:{population_str}*({country_str}),{feature_str}\n"

    # This is the code that is used to download the results of the clustering.
    st.text_area("Output", value=output_str)

    with io.BytesIO(output_str.encode()) as buffer:
        st.download_button(
            label=f"Download Results ({num_clusters} Clusters) as File",
            data=buffer.getvalue(),
            file_name=f"results_{num_clusters}_clusters.txt",
            mime="text/plain"
        )
