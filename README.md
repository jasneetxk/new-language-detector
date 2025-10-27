#  Multilingual Language Detector

The **Multilingual Language Detector** is a Streamlit-based NLP tool designed to automatically detect the language of any given text or document, including PDF and Word files. It can identify multiple languages within a single sentence, handle multilingual or mixed-language text seamlessly, and provide accurate predictions across 20 supported languages.

---

## Features

### Language Detection
This model is powered by a **TF-IDF (character n-gram) + Logistic Regression** pipeline. It was trained on balanced multilingual datasets covering 20 languages, ensuring robust and fair performance across diverse linguistic families. The detector supports both short and long text inputs, from single sentences to entire documents. It can process `.pdf` and `.docx` files and detect languages on a per-page or full-document basis. Additionally, it supports detection of **multiple languages within the same sentence** through token-level inference and voting.

### Streamlit Frontend
The frontend is built using **Streamlit**, providing a clean and interactive user experience. The sidebar displays essential information such as model details, supported languages, and project information. Users can paste text for instant classification, upload PDF or Word files for analysis, and test the model using preloaded language examples. For PDF documents, the interface also presents a page-by-page breakdown of detected languages.

---

## Model Training Overview

The **language detection model** was developed from scratch in `new_model.ipynb` using a TF-IDF and Logistic Regression pipeline. The training process combined text data from 20 different languages, ensuring a balanced and representative dataset. Text samples were preprocessed by converting to lowercase, removing punctuation, URLs, and emojis, and tokenizing into character n-grams. This preprocessing ensured consistent input representation and reduced noise in the model.

---

## Dataset and Feature Engineering

The dataset included sentences and short paragraphs for each of the 20 languages, ensuring uniform coverage and preventing bias. Each text sample was normalized and vectorized using the **TF-IDF (Term Frequency–Inverse Document Frequency)** technique, with the following parameters:
- `analyzer="char"`
- `ngram_range=(1, 3)`
- `max_features=10000`

By focusing on **character-level n-grams**, the model effectively captures short letter patterns and orthographic features unique to each language’s alphabet and writing system. This approach makes it highly effective even for short or code-switched sentences.

---

## Model Architecture

The classification layer uses **Multiclass Logistic Regression**, configured with the `"lbfgs"` solver and up to 1000 iterations. The model was trained with **balanced class weights** to ensure fair treatment of underrepresented languages. It predicts the probability distribution across all 20 supported languages and outputs the one with the highest confidence score.

This architecture achieves strong generalization performance across diverse inputs — from short text snippets to full-length documents — while remaining lightweight and fast for real-time inference within a web interface.

---

## Summary

In summary, the **Multilingual Language Detector** integrates an interpretable classical machine learning model with an intuitive Streamlit frontend. It provides reliable, fast, and accurate multilingual classification for text and documents without requiring deep learning infrastructure. Its use of TF-IDF character n-grams allows it to detect and differentiate languages with high precision, even in code-mixed or multilingual sentences.
