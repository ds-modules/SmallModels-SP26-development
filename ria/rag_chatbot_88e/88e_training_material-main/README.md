# 88e_training_material
Repo for training LLM tutor 

# 88E Training Material

This repository collects structured course materials from **Data 88E (Economic Models)**, prepared in multiple formats for use in **AI teaching experiments**, and notably this material is all in the public domain.  

The goal is two-fold:  
1. For **teachers** — to explore how course content can be packaged for Custom LLMs, Retrieval-Augmented Generation (RAG), or LoRA fine-tuning.  
2. For **students** — to learn hands-on how to build and test small language models using publicly accessible academic materials.

---

## Repository Structure

The same content is offered in **three parallel formats**:

1. **Individual Markdown Files within Folders**  
   - `F24Lec_MD/` — lecture notebooks (`lec01`, `lec02`, …).  
   - `F24Textbook_MD/` — textbook chapters (`00-Intro`, `01-Demand`, …).  
   - `F24LS_md/` — slide decks in Markdown.  
   - Each folder contains a `summary.yaml` describing its contents.  
   - Lecture Notebooks and Textbook Chapters are in nested folders by week for easier navigation.

2. **Zipped Folders**  
   - `.zip` archives of the above folders for easier upload to platforms that allow compressed input.  

3. **Concatenated Mega Files**  ( Eg for Gemini with 10 file limit)
   - One Markdown file per content type, merging all chapters/lectures/slides into a single file.  
   - Useful for services (Custom GPTs, Claude, Gemini) with strict file count limits.  
   - Each mega file begins with a summary for navigation.

---

## Metadata Files

- **`course_summary.yaml`** — overview of weeks, topics, and goals.  
- **`LecNB_summary.yaml`** — lecture notebook index with short blurbs.  
- **`week_to_readings.yaml`** — links schedule weeks to readings.  - the textbook wasnt lining up with the weeks properly
- **`summary.yaml`** — inside each folder, lightweight metadata describing scope.  

---

## How to Use This Repo

### For Teachers
- Upload **mega files** or **zips** to custom LLM platforms.  
- Use **YAML summaries** to improve chunking and retrieval quality.  
- Try different strategies (many small files vs. one large file) and compare performance.

### For Students
This repo doubles as a **sandbox for learning AI workflows**:

- **Custom LLMs:**  
  Experiment with uploading the mega files into tools like **Custom GPTs, Claude Projects, or Gemini Apps**. Compare how they answer student questions.  

- **RAG (Retrieval-Augmented Generation):**  
  Store the **individual Markdown files** in a vector database (e.g., Chroma, FAISS, Weaviate). Build a simple retrieval pipeline to answer questions from the textbook.  

- **LoRA Fine-Tuning:**  
  Pair textbook files (formal explanations) with lecture files (worked examples). Run a lightweight fine-tuning job on an open model (e.g. LLaMA-2, Mistral, Phi-3).  

---

## Why This Matters

- **Transparency:** These are publicly accessible, openly licensed academic materials.  
- **Pedagogy:** Students see how raw educational content becomes training data.  
- **Experimentation:** Both teachers and students can test tradeoffs (structured YAML vs. raw Markdown, folders vs. mega files, RAG vs. fine-tuning).  

---

## Next Steps

- Try building your own **course-aware study assistant**.  
- Compare retrieval quality with **different chunking strategies**.  
- Share your results with the teaching community.  

This repo is both a **curricular resource** and a **playground** for building and studying how language models learn.