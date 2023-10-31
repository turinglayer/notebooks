# Jupyter Notebooks using LLMs 

### 01. Binary Classification w/ SVM and Transformer-based Embeddings 

[[Notebook]](./01_binary_classification_svm.ipynb)
[[Open in Colab]](https://colab.research.google.com/github/turinglayer/notebooks/blob/main/01_binary_classification_svm.ipynb)

Tags: [binary-classification] [embeddings] [svm] [cohere] [openai] [tfidfvectorizer]

This notebook illustrates how to perform **binary text classification** with just a few hundred samples. It trains a basic **Support Vector Machine** with a collection of labeled financial sentences (375 training samples), and compares its accuracy with: 
- transformer-based embeddings using **[Cohere](https://docs.cohere.com/reference/embed)**.
- transformer-based embeddings using **[OpenAI](https://platform.openai.com/docs/api-reference/embeddings)**.
- frequency-based embeddings using **[TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)**

<p align="center">
  <img src="./static/embeddings.png">
</p>

### 02. Multiclass Classification w/ Random Forest and Transformer-based Embeddings 

[[Notebook]](./02_multi_classification_random_forest.ipynb)
[[Open in Colab]](https://colab.research.google.com/github/turinglayer/notebooks/blob/main/02_multi_classification_random_forest.ipynb)

Tags: [multiclass-classification] [embeddings] [hyperparameter-tuning] [random-forest] [cohere] [countvectorizer]

This notebook illustrates how to train a **random-forest** model with **hyperparameter tuning** for multiclass classification. It assesses the perfomance of combining said **random-forest** with:
- transformer-based embeddings using **[Cohere](https://docs.cohere.com/reference/embed)**.
- bag-of-words vectorizer using **[CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)**.

<p align="center">
  <img src="./static/randomforest.png">
</p>