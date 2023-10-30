# Jupyter Notebooks using LLMs 

### 01. Text Classification w/ Transformer-Based Embeddings

[[Notebook]](./01_classification_with_embeddings.ipynb)
[[Open in Colab]](https://colab.research.google.com/github/turinglayer/notebooks/blob/main/01_classification_with_embeddings.ipynb)


This notebook illustrates how to perform **binary text classification** with just a few hundred samples. It trains a simple **Support Vector Machine** with a collection of labeled financial sentences (375 training samples), and compares its accuracy with: 
- transformer-based embeddings using **[Cohere](https://docs.cohere.com/reference/embed)**.
- transformer-based embeddings using **[OpenAI](https://platform.openai.com/docs/api-reference/embeddings)**.
- frequency-based embeddings using **TfidfVectorizer**

<p align="center">
  <img src="./static/embeddings.png">
</p>