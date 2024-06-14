import xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer
import json
from numpyencoder import NumpyEncoder
import warnings
warnings.simplefilter("ignore", category=FutureWarning)
import numpy as np

""" Step 1: Parse the XML Sitemap en-gb """

def url_mappings_1():
    for url in root1.findall('ns:url', namespaces):
            loc1 = url.find('ns:loc', namespaces).text
            content1 = url.find('ns:content', namespaces).text if url.find('ns:content', namespaces) is not None else ""
            locale1 = loc1.split('/')[3]  # Assuming the locale is the first segment after the domain
            path1 = loc1.split('/', 4)[-1]  # Get path after the locale segment

            if locale1 not in url_mappings1:
                url_mappings1[locale1] = {}
            url_mappings1[locale1][path1] = content1


tree1 = ET.parse('en-gb_sitemap.xml')
root1 = tree1.getroot()
namespaces = {'ns': 'https://www.sitemaps.org/schemas/sitemap/0.9'}
url_mappings1 = {}
url_mappings_1()

# Load the multilingual model
model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

# Generate embeddings
embeddings_dict1 = {}
for locale1, paths1 in url_mappings1.items():
    embeddings_dict1[locale1] = {path1: model.encode(content1) for path1, content1 in paths1.items()}

# Save embeddings to a file en-gb
with open('embeddings1.json', 'w') as f:
    json.dump(embeddings_dict1, f, cls=NumpyEncoder)

""" Step 2: Parse the XML Sitemap de-de """

def url_mappings_2():
    for url in root2.findall('ns:url', namespaces):
            loc2 = url.find('ns:loc', namespaces).text
            content2 = url.find('ns:content', namespaces).text if url.find('ns:content', namespaces) is not None else ""
            locale2 = loc2.split('/')[3]  # Assuming the locale is the first segment after the domain
            path2 = loc2.split('/', 4)[-1]  # Get path after the locale segment

            if locale2 not in url_mappings2:
                url_mappings2[locale2] = {}
            url_mappings2[locale2][path2] = content2

tree2 = ET.parse('de-de_sitemap.xml')
root2 = tree2.getroot()
namespaces = {'ns': 'https://www.sitemaps.org/schemas/sitemap/0.9'}
url_mappings2 = {}
url_mappings_2()

# Load the multilingual model
model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

embeddings_dict2 = {}
for locale2, paths2 in url_mappings2.items():
    embeddings_dict2[locale2] = {path2: model.encode(content2) for path2, content2 in paths2.items()}

# Save embeddings to a file en-gb
with open('embeddings2.json', 'w') as f:
    json.dump(embeddings_dict2, f, cls=NumpyEncoder)

""" Step 3: Compute Similarity and Find Equivalent URL """

def cosine_similarity(emb1, emb2):
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

def find_equivalent_url(current_url, current_locale, target_locale):

    # Extract path from current URL
    current_path = current_url.split(current_locale)[1].lstrip('/')

    # Get current embedding
    current_embedding = embeddings_dict1[current_locale][current_path]

    # Find the most similar URL in the target locale
    max_similarity = -1
    equivalent_url = None

    for path, target_embedding in embeddings_dict2[target_locale].items():
        similarity = cosine_similarity(current_embedding, target_embedding)
        if similarity > max_similarity:
            max_similarity = similarity
            equivalent_url = f"https://www.asafe.com/{target_locale}/{path}"
            
    return equivalent_url

# Example usage
current_url = "https://www.asafe.com/en-gb/contact/"
current_locale = "en-gb"
target_locale = "de-de"

with open('embeddings2.json', 'r') as f:
    embeddings_dict2 = json.load(f)
with open('embeddings1.json', 'r') as f:
    embeddings_dict1 = json.load(f)

equivalent_url = find_equivalent_url(current_url, current_locale, target_locale)
print(f"Equivalent URL: {equivalent_url}")