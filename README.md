# Assignment 3 - Shannon’s Information Theory & Text Generation  
**Course:** CSE 510 - Basics of Artificial Intelligence  
**Student:** Aryaa Paresh Deshpande  
**University at Buffalo**

---

## Overview
This project re-creates Claude Shannon’s 1948 demonstration of statistical language generation using *n-gram* models.  
It involves analyzing text data from three authors - **Jane Austen**, **Mark Twain**, and **Arthur Conan Doyle** - to:
1. Compute n-gram frequency tables  
2. Analyze and visualize sentence statistics  
3. Generate new text using Markov models of varying order  
4. Provide a unified command-line interface for all tasks  
5. Reflect on results through a written report

---

## Environment Setup

### 1. Clone the Repository
```
git clone https://github.com/aryaa-deshpande/BOAI-Assignment-3.git

cd BOAI-Assignment-3
```

### 2.Create a Virtual Environment
```
python3 -m venv myenv

source myenv/bin/activate       
```

### 3.Install Dependencies
```
pip install -r requirements.txt
```


---

## How to Run the Project

The entire pipeline can be run through a single interface:
src/shannon_gen.py

Each stage corresponds to one part of the assignment.

---

### Part 1 - Frequency Analysis

Clean and process text files, and generate 1-, 2-, and 3-gram frequency tables.
```
python3 src/shannon_gen.py analyze --author austen
python3 src/shannon_gen.py analyze --author twain
python3 src/shannon_gen.py analyze --author doyle
```
Outputs:
```
	•	data/freq_tables/austen_word_*.json
	•	data/freq_tables/twain_word_*.json
	•	data/freq_tables/doyle_word_*.json
```

---

### Part 2 - Statistical Analysis and Visualization

Compute sentence statistics and create visualizations for each author.
```
python3 src/shannon_gen.py visualize --author austen
python3 src/shannon_gen.py visualize --author twain
python3 src/shannon_gen.py visualize --author doyle
```
Outputs:
```
	•	Sentence length histograms
	•	Top 3-gram frequency bar plots
```
All saved in the /outputs/cli_screenshots folder as .png files.

---

### Part 3 - Text Generation

Generate new text using word- or character-based Markov models.
```
python3 src/shannon_gen.py generate --author austen --level word-3 --length 50
python3 src/shannon_gen.py generate --author twain --level word-3 --length 50
python3 src/shannon_gen.py generate --author doyle --level word-3 --length 50
```
Example Output (Austen, word-3):
```
and they took place they must have occasion to her of georgiana’s delight in her interest…
```
The text is printed directly in the terminal.

---

### Part 4 - Unified CLI

All tasks above are integrated into shannon_gen.py.

Use the following structure for any author and operation:
```
python3 src/shannon_gen.py <command> --author <austen|twain|doyle> [additional arguments]
```
Examples:
```
python3 src/shannon_gen.py analyze --author austen
python3 src/shannon_gen.py visualize --author twain
python3 src/shannon_gen.py generate --author doyle --level word-3 --length 50
```

---

### Part 5 - Report and Reflection

The detailed written analysis is located in:
```
report/Analysis_Report.md
report/Analysis_Report.pdf
```
It includes discussion of:
```
	•	Text quality across model orders
	•	Author distinctiveness
	•	Anchor word integration
	•	Connections to modern language models
	•	Shannon’s key insights
```

---

### Project Structure
```
BOAI-Assignment-3/
├── data/
│   ├── freq_tables/
│   │   ├── austen_word_1-gram.json
│   │   ├── doyle_word_3-gram.json
│   │   ├── twain_word_3-gram.json
│   ├── austen_pride_prejudice.txt
│   ├── doyle_sherlock_holmes.txt
│   └── twain_tom_sawyer.txt
├── outputs/
│   ├── cli_screenshots/
│   │   ├── austen.png
│   │   ├── twain.png
│   │   └── doyle.png
│   ├── *_sentence_length_hist.png
│   ├── *_word_top3grams.png
│   └── *_character_top3grams.png
├── report/
│   ├── Analysis_Report.md
│   └── Analysis_Report.pdf
├── src/
│   ├── analyze.py
│   ├── analyze_stats.py
│   ├── generator.py
│   └── shannon_gen.py
├── starter_preprocess.py
├── assignment3_grading_tester.py
├── requirements.txt
└── README.md
```

