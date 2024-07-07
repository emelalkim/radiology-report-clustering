## Radiology Report Clustering

This project aims to cluster de-identified radiology reports based on their similarity. Report similarity is defined in terms of their clinical findings. The main objective is to group similar reports together for further analytics, such as understanding the mix of findings for an individual radiologist or a practice. The project utilizes Natural Language Processing (NLP) techniques to preprocess the reports, compute similarities, and apply clustering algorithms.

### Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Approach](#approach)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/emelalkim/radiology-report-clustering.git
   cd radiology-report-clustering
   ```

2. **Install Dependencies**:
   ```bash
   pip install pandas scikit-learn nltk matplotlib pandas

   ```
   or (for MAC python 3)
   ```bash
   pip3 install pandas scikit-learn nltk matplotlib pandas 

   ```

3. **Ensure RadLex Terms CSV is Available**:
   Make sure the `RADLEX_termsonly.csv` file is in the project directory.

### Usage

1. **Prepare the Data**:
   Ensure you have the de-identified radiology reports in a CSV file (e.g., `reports.csv`). Each report should have a `clean_report` column containing the report text.

2. **Run the Script**:
   ```bash
   python similarity_analysis.py
   ```

3. **Outputs**:
   - Clustering results: `clustering_results_dbscan.csv`
   - Cluster summary: `cluster_summary_dbscan.txt`

### Project Structure

- `similarity_analysis.py`: Main script to preprocess reports, compute similarities, find optimal DBSCAN parameters, and perform clustering.
- `RADLEX_termsonly.csv`: CSV file containing RadLex terms and synonyms for medical term normalization.

### Approach

1. **Data Preparation**:
   - Extract the FINDINGS section from each radiology report.
   - Normalize medical terms using RadLex dictionary.
   - Tag measurements and preprocess text to remove stopwords and special characters.

2. **Feature Extraction**:
   - Compute TF-IDF vectors for the preprocessed findings sections.

3. **Similarity Calculation**:
   - Calculate the cosine distance matrix between TF-IDF vectors.

4. **Parameter Optimization**:
   - Plot the k-distance graph to find the optimal epsilon value for DBSCAN. This method can be used to plot the graph to see where thw knee is. 

5. **Clustering**:
   - Apply DBSCAN using the optimal epsilon value and cosine distance matrix.

6. **Results Analysis**:
   - Generate and save clustering results and summaries.
   - Display samples of the most similar and most dissimilar reports.

### Results

- The script outputs the clustering results in `clustering_results_dbscan.csv`, showing which cluster each report belongs to.
- A summary of each cluster's top keywords is saved in `cluster_summary_dbscan.txt`.
- The script also prints the most similar and most dissimilar reports based on cosine similarity.

### Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your improvements.

### License

This project is licensed under the MIT License. See the `LICENSE` file for details.
