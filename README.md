# Smart Medical Chatbot using Open-Source LLM Tools

Welcome to the **Smart Medical Chatbot** project!.

---

## Project Overview

In this project, we build a smart, interactive chatbot designed to assist with medical queries. It uses:
- **HuggingFace Embeddings** for semantic understanding
- **Faiss (CPU)** for efficient vector search
- **Mistral** as the conversational Large Language Model (LLM)
- **Streamlit** for creating an intuitive and user-friendly web interface

Whether you're a beginner or a tech enthusiast, this guide walks you through each step to create a robust conversational AI solution for healthcare.

---

## Tech Stack

| Tool/Library         | Purpose                                          |
|----------------------|--------------------------------------------------|
| `Pipenv`             | Virtual environment & dependency management      |
| `langchain`          | Orchestration of LLM pipelines                   |
| `langchain_community`| Access to community-built LangChain integrations|
| `langchain_huggingface` | Embedding support via HuggingFace Transformers|
| `huggingface_hub`    | HuggingFace model and dataset integration       |
| `faiss-cpu`          | Local vector similarity search engine           |
| `pypdf`              | For processing medical documents (PDF)          |
| `streamlit`          | Interactive frontend for chatbot UI             |

---

## Setup Instructions

### 1. Prerequisite: Install Pipenv

Make sure you have **Pipenv** installed.  
👉 [Pipenv Installation Guide](https://pipenv.pypa.io/en/latest/installation.html)

---

### 2. Clone the Repository

```bash
git clone https://github.com/pavangv12/smart-medical-chatbot.git
cd smart-medical-chatbot
````

---

### 3. Install Project Dependencies

Use Pipenv to install all required packages:

```bash
pipenv install langchain langchain_community langchain_huggingface faiss-cpu pypdf
pipenv install huggingface_hub
pipenv install streamlit
```

---

### 4. Run the Chatbot

Activate the virtual environment and launch the app:

```bash
pipenv shell
streamlit run app.py
```

---

## What You'll Learn

* How to use **semantic embeddings** to represent text
* Store and query information using **vector databases**
* Build a conversation pipeline using **open-source LLMs**
* Design a responsive chatbot interface with **Streamlit**

---

## Acknowledgements

* [HuggingFace](https://huggingface.co/) for making cutting-edge models accessible
* [LangChain](https://www.langchain.com/) for simplifying LLM integration
* [Faiss](https://github.com/facebookresearch/faiss) for efficient vector search
* [Streamlit](https://streamlit.io/) for rapid prototyping

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

Made with 💙 by **\[Pavan]**

```
