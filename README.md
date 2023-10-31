# Jupyter Notebooks using LLMs 

### 01. Binary Classification w/ SVM and Transformer-Based Embeddings

[[Notebook]](./01_binary_classification_svm.ipynb)
[[Open in Colab]](https://colab.research.google.com/github/turinglayer/notebooks/blob/main/01_binary_classification_svm.ipynb)


This notebook illustrates how to perform **binary text classification** with just a few hundred samples. It trains a basic **Support Vector Machine** with a collection of labeled financial sentences (375 training samples), and compares its accuracy with: 
- transformer-based embeddings using **[Cohere](https://docs.cohere.com/reference/embed)**.
- transformer-based embeddings using **[OpenAI](https://platform.openai.com/docs/api-reference/embeddings)**.
- frequency-based embeddings using **[TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)**

<p align="center">
  <img src="./static/embeddings.png">
</p>