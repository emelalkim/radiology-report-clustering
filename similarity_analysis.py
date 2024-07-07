import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

class ReportAnalyzer:
    def __init__(self):
        # Download stop words
        nltk.download('stopwords')
        self.stops = set(stopwords.words("english"))
        self.measurement_regex = self.prepare_measurement_regex()
        self.medical_terms, self.synonym_dict = self.load_medical_terms()

    # Define a function to extract the FINDINGS section from the clean report
    def extract_findings(self, clean_report):
        findings_match = re.search(r'(FINDINGS|Findings):(.*?)(IMPRESSION:|Impression:|$)', clean_report, re.DOTALL)
        if findings_match:
            findings = findings_match.group(2).strip()
        else:
            findings = ''
        return findings.lower()

    # Prepare the measurement regex
    def prepare_measurement_regex(self):
        x = r"(\d+\.( )?\d+" + "|\d+( )?\.\d+" + "|\.\d+|\d+) *"
        by = r"( )?(by|x)( )?"
        cm = r"([\- ](mm|cm|millimeter(s)?|centimeter(s)?)(?![a-z/]))"
        x_cm = r"((" + x + " *(to|\-) *" + cm + ")" + "|(" + x + cm + "))"
        xy_cm = r"((" + x + cm + by + x + cm + ")" + "|(" + x + by + x + cm + ")" + "|(" + x + by + x + "))"
        xyz_cm = r"((" + x + cm + by + x + cm + by + x + cm + ")" + "|(" + x + by + x + by + x + cm + ")" + "|(" + x + by + x + by + x + "))"
        m = r"((" + xyz_cm + ")" + "|(" + xy_cm + ")" + "|(" + x_cm + "))"
        return re.compile(m)

    # Format and tag the measurement
    def replace_measurement_regex(self, match):
        text = match.group()
        text = '_'.join(text.lstrip().rstrip().split())
        text = text + '|MEASUREMENT'
        return text

    # Load RadLex medical terms from a CSV file
    def load_medical_terms(self):
        radlex_file_path = './RADLEX_termsonly.csv'  # Update with the correct file path
        radlex_df = pd.read_csv(radlex_file_path)
        medical_terms = set(radlex_df['Preferred Label'].dropna().unique())
        # Create a synonym dictionary
        synonym_dict = {}
        
        for _, row in radlex_df.iterrows():
            preferred_name = str(row['Preferred Label'])
            synonyms = str(row['Synonyms'])
            if 'RID' not in preferred_name and 'RID' not in synonyms and pd.notna(preferred_name) and pd.notna(synonyms):
                for synonym in synonyms.split('|'):
                    synonym_dict[synonym.strip()] = preferred_name.replace(' ', '_')
        
        return medical_terms, synonym_dict

    # Preprocess the text by removing stopwords, tagging measurements, and converting medical terms to underscored text
    def preprocess_text(self, text):
        words = text.split()
        # Remove stop words
        tokens = [word for word in words if word not in self.stops]
        clean_text = ' '.join(tokens)

        # Tag measurements
        clean_text = re.sub(self.measurement_regex, self.replace_measurement_regex, clean_text)

        # Replace medical terms and synonyms with underscored text
        tokens = clean_text.split()
        processed_tokens = []
        for token in tokens:
            # if it is a synonym, use the radlex medical term instead of the synonym to normalize the data
            if token in self.synonym_dict:
                processed_tokens.append(self.synonym_dict[token].replace(' ', '_'))
            elif token in self.medical_terms:
                processed_tokens.append(token.replace(' ', '_'))
            else:
                processed_tokens.append(token)
        
        clean_text = ' '.join(processed_tokens)

        # Clean up 
        clean_text = re.sub(r'[^a-zA-Z0-9_|\s]', '', clean_text)
        return clean_text

    # Custom tokenizer to ensure measurements and medical terms are tagged correctly
    def custom_tokenizer(self, text):
        tokens = text.split()
        tokens = ['_measurement_' if '|MEASUREMENT' in token else token for token in tokens]
        return tokens

    # Read the report data from the CSV and preprocess
    def read_and_prep_data(self, file_path):
        data = pd.read_csv(file_path)
        data['findings_text'] = data['clean_report'].apply(self.extract_findings)
        data['prepped_findings_text'] = data['findings_text'].apply(self.preprocess_text)
        print('Data cleaned and tagged. A sample report looks like this: \n', data['prepped_findings_text'][5])
        return data

    # Calculate TF-IDF matrix
    def calculate_tfidf_matrix(self, reports):
        vectorizer = TfidfVectorizer(tokenizer=self.custom_tokenizer)
        tfidf_matrix = vectorizer.fit_transform(reports['prepped_findings_text'])
        return tfidf_matrix

    # Compute cosine distance matrix
    def calculate_cosine_distance(self, tfidf_matrix):
        cosine_dist = cosine_distances(tfidf_matrix)
        return cosine_dist

    # Get the top keywords for the clusters
    def get_top_keywords(self, texts, num_keywords=10):
        all_text = ' '.join(texts)
        words = self.custom_tokenizer(self.preprocess_text(all_text))
        word_freq = pd.Series(words).value_counts().head(num_keywords)
        return word_freq.index.tolist()

    # Find the clusters and prepare cluster summary
    def cluster(self, reports, cosine_dist):
        # Apply DBSCAN clustering with cosine distance
        dbscan = DBSCAN(metric='precomputed', eps=0.58, min_samples=20)
        reports['cluster'] = dbscan.fit_predict(cosine_dist)

        cluster_summary = {}
        for cluster in sorted(reports['cluster'].unique()):
            if cluster != -1:  # Skip noise points
                cluster_texts = reports[reports['cluster'] == cluster]['findings_text']
                cluster_summary[f'Cluster {cluster}'] = f'Size: {len(cluster_texts)}; Top Keywords: {", ".join(self.get_top_keywords(cluster_texts, 10))}'

        # Save the clustering results and summary to files
        reports.to_csv('./clustering_results_dbscan.csv', index=False)

        with open('./cluster_summary_dbscan.txt', 'w') as f:
            for cluster, info in cluster_summary.items():
                f.write(f"{cluster}: {info}\n")

        # Display the clustering results and summary
        print("Clustering results saved to './clustering_results_dbscan.csv'.")
        print("Cluster summary saved to './cluster_summary_dbscan.txt'.")

    # Display samples of similar and different reports
    def display_samples(self, reports, cosine_dist):
        # Convert cosine distances to similarities
        cosine_sim = 1 - cosine_dist

        # Find the most similar and dissimilar reports
        most_similar = (None, None)
        most_dissimilar = (None, None)
        max_similarity = -1
        min_similarity = 1

        num_reports = cosine_sim.shape[0]
        for i in range(num_reports):
            for j in range(i + 1, num_reports):
                if cosine_sim[i, j] > max_similarity and cosine_sim[i, j]<1:
                    max_similarity = cosine_sim[i, j]
                    most_similar = (i, j)
                if cosine_sim[i, j] < min_similarity and cosine_sim[i, j]>=0:
                    min_similarity = cosine_sim[i, j]
                    most_dissimilar = (i, j)

        print("Most Similar Reports:")
        print("Report 1:", reports.iloc[most_similar[0]]['clean_report'])
        print("Report 2:", reports.iloc[most_similar[1]]['clean_report'])

        print("\nMost Dissimilar Reports:")
        print("Report 1:", reports.iloc[most_dissimilar[0]]['clean_report'])
        print("Report 2:", reports.iloc[most_dissimilar[1]]['clean_report'])

    # Plot k-distance graph to find optimal eps value for DBSCAN
    # This method can be used in Jupiter notebook. It was used to plot the k-distance graph 
    #   and decide with eps to use
    def plot_k_distance_graph(self, vectors, k=5):
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors_fit = neighbors.fit(vectors)
        distances, indices = neighbors_fit.kneighbors(vectors)
        distances = np.sort(distances, axis=0)
        distances = distances[:, 1]
        plt.plot(distances)
        plt.xlabel('Points sorted by distance')
        plt.ylabel('k-distance')
        plt.title('k-distance Graph for DBSCAN')
        plt.show()


def main():
    my_report_analyzer = ReportAnalyzer()
    # Read the csv and preprocess the reports
    reports = my_report_analyzer.read_and_prep_data('./reports.csv')

    # Compute the TF-IDF matrix
    tfidf_matrix = my_report_analyzer.calculate_tfidf_matrix(reports)
    print('TFIDF matrixes are created successfully')
    
    # Compute the cosine distance matrix
    cosine_dist = my_report_analyzer.calculate_cosine_distance(tfidf_matrix)
    print('Cosine distances are calculated successfully')
    print(cosine_dist)
   
    # Display samples of similar and different reports
    my_report_analyzer.display_samples(reports, cosine_dist)

    # Plot k-distance graph
    my_report_analyzer.plot_k_distance_graph(tfidf_matrix.toarray(), k=5)

    # Find the clusters using DBSCAN with cosine distance
    my_report_analyzer.cluster(reports, cosine_dist)


if __name__ == "__main__":
    main()
