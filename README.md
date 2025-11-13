# EPUB2Code
Convert any `.epub` book into a fake `.py` code file that you can read inside your favorite IDE.

This tool transforms epub text into code-like text using code structures (functions, loops, classes, try/except blocks, etc.) so that the entire book **looks like source code**, but keeps the real text intact and readable.  
Intented for reading at work, in class, or whenever you want to fulfill your dream of reading in an IDE.

---

## Features

- Converts any `.epub` file into a `.py` file  
- Preserves sentences using:
    - variable assignments  
   - comments  
- Wraps the text inside pseudo-code structures:
  - functions  
  - classes  
  - if/else blocks  
  - for loops  
  - try/except blocks 
- Optional `--seed` argument to keep the pseudo-random structure consistent  
- Works with any EPUB that contains extractable text

---

## Tutorial

You need Python 3.9 or newer.

Install dependencies:

```bash
pip install ebooklib beautifulsoup4
```
Transform your EPUB to a py file:

```bash
python epub2code.py "my_book.epub" -o "my_book_as_code.py" --seed 42
```

---

## Output

<img width="1440" height="381" alt="image" src="https://github.com/user-attachments/assets/ab17d280-bd60-49ce-9905-eadfc39e3b04" />


- For future possible improvements:
  - Add more fake code so the text is not as dominant. ✅
  - Allow choosing a value for text to code ratio. ✅
  - Split output into various files to manage line amount.
  - Avoid treating abbreviations or prefixes as whole sentences (e.g. Mr.) ✅

Feel free to suggest or contribute anything.

was vibecoded idgaf
