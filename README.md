# AI-Powered Academic Research Assistant
The **AI-Powered Academic Research Assistant** is a comprehensive, online chatbot designed to assist students and researchers in crafting high-quality academic works. This application leverages advanced AI technology to generate, refine, and analyze academic writing, allowing users to maintain formal style, grammatical precision, and logical coherence throughout their work.

## Features

#### The main features of the AI-Powered Academic Research Assistant include:

- **Text Generation**: The chatbot assists in generating contextually relevant academic text based on prompts or partial content already provided by the user. This helps users build on their ideas and expand on specific topics.
- **Grammar Correction**: Uses a *T5 Base Grammar Correction Model* to ensure grammatical accuracy and clarity in the user's writing.
- *Formal Style Analysis*: Evaluates the academic tone and formal style of the text, ensuring adherence to scholarly writing conventions via a specialized *Style Transformer*.

## Technical Architecture

The application’s architecture is built to provide fast, reliable, and high-quality academic assistance using the following components:

### Language Models and Components
- [**Qwen2.5 1.5B Instruct**](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct): The main large language model (LLM) used for text generation, capable of producing nuanced and detailed academic content.
- [**T5 Base Grammar Correction Model**](https://huggingface.co/vennify/t5-base-grammar-correction): Ensures grammatical correctness by identifying and correcting grammar issues in the user's input text.
- [**Style Transformer**](https://github.com/PrithivirajDamodaran/Styleformer): Preserves and enhances the formal academic style, adapting responses to align with scholarly writing conventions.
- **RAG (Retrieval-Augmented Generation) System**: A vector database that consists of a carefully curated collection of academic works, which serves as a knowledge base for the LLM. This dataset provides the model with a rich source of context and reference, ensuring responses are both accurate and relevant ([Base Dataset](https://huggingface.co/datasets/somosnlp-hackathon-2022/scientific_papers_en/viewer/default/train?row=0)).
- **Routing Approach**: Optimizes performance by directing specific types of user queries to the most suitable processing components within the application.

### Front-End Interface
The front end of the application is designed for intuitive interaction, using:

- **Gradio API**: Simplifies the creation of an interactive and user-friendly front-end.
- **Deployment on Hugging Face Spaces**: Makes the app easily accessible online as a Hugging Face demo, enabling users to experience the chatbot’s features with minimal setup.

## Installation and Usage
To use the AI-Powered Academic Research Assistant you can: 
1. **Try the application online** following the [Demo link](link) and accessing the deployed app via Hugging Face Spaces.

2. **Run local instance**:
```bash
git clone https://github.com/Pupolina7/ResearchAssistant
```
```bash
pip install -r requirements.txt
```
```bash
python main.py
```

## Future Enhancements
To expand the functionality and user experience of the **AI-Powered Academic Research Assistant**, the following features are planned for future updates:

- **'Stop Generation' Button**: Introduce a button that allows users to halt the text generation process mid-response, offering more control over the interaction.
- **File Content Writing**: Enable the assistant to directly write generated or improved content into user-uploaded files, making it easier to incorporate edits without manual copy-pasting.
- **Markdown Support**: Enhance the app’s capabilities to interpret and generate Markdown (MD) formatted content, ensuring compatibility with popular text editors and document formats.
- **LaTeX Support**: Add support for LaTeX, enabling researchers in fields like mathematics, physics, and engineering to input and output complex equations and scientific notations seamlessly.
- **Improvement Hints**: Provide contextual hints and suggestions for enhancing the clarity, coherence, or academic rigor of the user’s text, offering actionable feedback to elevate the quality of writing.





