#!/usr/bin/env python3
"""
epub2code.py
Convert an EPUB book into a Python code file that simulates reading the book through code constructs.
"""

import os
import re
import json
import random
import argparse
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

# -----------------------
# Utilities
# -----------------------

SENT_SPLIT_RE = re.compile(r'(?<=[\.\?\!])\s+(?=[A-Z0-9"“‘\(\[])')
ABR_DOT = "•"

def mask_abbrev_periods(text: str) -> str:
    t = text
    t = re.sub(r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc)\.", lambda m: m.group(0).replace(".", ABR_DOT), t)
    t = re.sub(r"\b(?:e\.g|i\.e)\.", lambda m: m.group(0).replace(".", ABR_DOT), t)
    t = re.sub(r"\bPh\.D\.", lambda m: m.group(0).replace(".", ABR_DOT), t)
    t = re.sub(r"\bM\.Sc\.", lambda m: m.group(0).replace(".", ABR_DOT), t)
    t = re.sub(r"\bB\.Sc\.", lambda m: m.group(0).replace(".", ABR_DOT), t)
    t = re.sub(r"\bM\.A\.", lambda m: m.group(0).replace(".", ABR_DOT), t)
    t = re.sub(r"\bB\.A\.", lambda m: m.group(0).replace(".", ABR_DOT), t)
    t = re.sub(r"\bU\.S\.(?:A\.)?", lambda m: m.group(0).replace(".", ABR_DOT), t)
    t = re.sub(r"\bU\.K\.", lambda m: m.group(0).replace(".", ABR_DOT), t)
    t = re.sub(r"\b(?:[A-Z]\.){2,3}", lambda m: m.group(0).replace(".", ABR_DOT), t)
    t = re.sub(r"(?:\b[A-Z]\.\s*){2,}", lambda m: m.group(0).replace(".", ABR_DOT), t)
    t = re.sub(r"\b[A-Z]\.(?=\s+[A-Z])", lambda m: m.group(0).replace(".", ABR_DOT), t)
    return t

def clean_whitespace(s: str) -> str:
    s = s.replace('\r', ' ')
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def split_into_sentences(text: str) -> list:
    """Split text into sentences using a practical regex.
    Keeps punctuation at end. Filters out trivial fragments."""
    text = clean_whitespace(text)
    if not text:
        return []
    masked = mask_abbrev_periods(text)
    parts = SENT_SPLIT_RE.split(masked)
    # Remove extremely short fragments
    sentences = [p.replace(ABR_DOT, ".").strip() for p in parts if len(p.strip()) > 1]
    return sentences

def escape_str(s: str) -> str:
    """Return a safe JSON-style string literal for embedding in code."""
    return json.dumps(s, ensure_ascii=False)

class IdGen:
    def __init__(self, prefix='x'):
        self.prefix = prefix
        self.counter = 0
    def next(self):
        self.counter += 1
        return f"{self.prefix}{self.counter:03d}"

# -----------------------
# EPUB reading (spine order)
# -----------------------

def extract_chapters_from_epub(path: str) -> list:
    """
    Extracts visible text from the EPUB following the spine order.
    Returns list of chapter-like blocks (strings).
    """
    book = epub.read_epub(path)
    chapters = []
    current = []

    # book.spine is list of (idref, linear?) tuples
    for idref, _ in book.spine:
        item = book.get_item_with_id(idref)
        if item is None:
            continue
        if item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue

        soup = BeautifulSoup(item.get_content(), 'html.parser')

        # Remove scripts/styles (if present)
        for tag in soup(['script', 'style']):
            tag.decompose()

        # Extract text lines, preserving order
        text = soup.get_text(separator='\n')
        lines = [clean_whitespace(line) for line in text.splitlines()]
        lines = [line for line in lines if line]

        if not lines:
            continue

        # Heuristic: if the item contains a heading that likely marks a new chapter,
        # start a new chapter block. We'll look for "chapter" or headings.
        lower_lines = " ".join(lines).lower()
        if ('chapter' in lower_lines) or any(len(line) < 80 and line.isupper() for line in lines[:3]):
            # if current has content, push it first
            if current:
                chapters.append("\n\n".join(current))
                current = []
            current.extend(lines)
        else:
            current.extend(lines)

    if current:
        chapters.append("\n\n".join(current))

    return chapters

# -----------------------
# Emitters for constructs
# -----------------------

def build_header(ch_num: int, title_hint: str = None) -> str:
    head = []
    head.append(f"def chapter_{ch_num}():")
    head.append(f"    #────────────────────────────────────────────────────────")
    title = f"Chapter {ch_num}"
    if title_hint:
        title = f"{title}: {title_hint}"
    head.append(f"    # {title}")
    head.append(f"    #────────────────────────────────────────────────────────")
    head.append("")  # blank line
    return "\n".join(head)

def emit_comment_lines(sentences: list, indent: int = 4) -> str:
    indent_s = " " * indent
    out = []
    for s in sentences:
        out.append(f"{indent_s}# {s}")
    out.append("")
    return "\n".join(out)

def emit_string_vars(sentences: list, idgen: IdGen, indent: int = 4) -> str:
    indent_s = " " * indent
    out = []
    for s in sentences:
        var = idgen.next()
        out.append(f"{indent_s}{var} = {escape_str(s)}")
    out.append("")
    return "\n".join(out)

def emit_print_lines(sentences: list, indent: int = 4) -> str:
    indent_s = " " * indent
    out = []
    for s in sentences:
        out.append(f"{indent_s}print(f{escape_str(s)})")
    out.append("")
    return "\n".join(out)

def emit_if_else(comment_sent, true_sents, false_sents, idgen: IdGen, indent: int = 4) -> str:
    indent_s = " " * indent
    s = []
    s.append(f"{indent_s}if True:")  # decorative
    for cs in comment_sent:
        s.append(f"{indent_s}    # {cs}")
    for ts in true_sents:
        var = idgen.next()
        s.append(f"{indent_s}    {var} = {escape_str(ts)}")
    if false_sents:
        s.append(f"{indent_s}else:")
        # else: emit one comment and/or one string variable depending on count
        for fs in false_sents:
            # alternate comment and var if many
            s.append(f"{indent_s}    # {fs}")
    s.append("")
    return "\n".join(s)

def emit_for_loop(loop_comment_sents, loop_string_sents, fake_ops_count:int, idgen: IdGen, indent: int = 4) -> str:
    indent_s = " " * indent
    out = []
    out.append(f"{indent_s}for _ in range(1):")
    for cs in loop_comment_sents:
        out.append(f"{indent_s}    # {cs}")
    for ss in loop_string_sents:
        var = idgen.next()
        out.append(f"{indent_s}    {var} = {escape_str(ss)}")
    for i in range(fake_ops_count):
        out.append(f"{indent_s}    _tmp_{i} = _ + {i}")
    out.append("")
    return "\n".join(out)

def emit_match_case(cases: list, idgen: IdGen, indent: int = 4) -> str:
    """
    cases: list of (label, [sent1, sent2?]) where each case may have 1-2 sentences.
    """
    indent_s = " " * indent
    out = []
    out.append(f"{indent_s}match 'state':")
    for label, sentlist in cases:
        out.append(f"{indent_s}    case {escape_str(label)}:")
        if len(sentlist) >= 1:
            out.append(f"{indent_s}        # {sentlist[0]}")
        if len(sentlist) >= 2:
            var = idgen.next()
            out.append(f"{indent_s}        {var} = {escape_str(sentlist[1])}")
    out.append("")
    return "\n".join(out)

def emit_try_except(try_sents, except_sents, idgen: IdGen, indent: int = 4) -> str:
    """
    Emits a try/except block where:
    - try_sents are split into comment-only and variable-only sentences
    - No sentence is repeated
    - except_sents remain comments only
    """
    indent_s = " " * indent
    out = []

    # Split try_sents so first half → comments, second half → variables
    # This avoids duplication
    half = max(1, len(try_sents) // 2)
    comment_sents = try_sents[:half]
    var_sents = try_sents[half:]

    out.append(f"{indent_s}try:")

    # Emit distinct comment sentences
    for ts in comment_sents:
        out.append(f"{indent_s}    # {ts}")

    # Emit distinct variable assignments (no repeats)
    for ts in var_sents:
        var = idgen.next()
        out.append(f"{indent_s}    {var} = {escape_str(ts)}")

    # EXCEPT block
    out.append(f"{indent_s}except Exception as e:")
    for es in except_sents:
        out.append(f"{indent_s}    # {es}")

    out.append("")
    return "\n".join(out)


def emit_small_class(class_name: str, sentences:list, idgen: IdGen, indent:int = 4) -> str:
    indent_s = " " * indent
    out = []
    out.append(f"{indent_s}class {class_name}:")
    if not sentences:
        out.append(f"{indent_s}    pass")
    else:
        for i, snt in enumerate(sentences):
            if i % 2 == 0:
                var = idgen.next()
                out.append(f"{indent_s}    {var} = {escape_str(snt)}")
            else:
                out.append(f"{indent_s}    # {snt}")
    out.append("")
    return "\n".join(out)

# -----------------------
# Sequential chapter builder (preserves book order)
# -----------------------

def build_chapter_function_sequential(sent_list: list, ch_idx: int, rng: random.Random, text_ratio: float = 0.5) -> str:
    """
    Build a chapter using sequential consumption of sentences (preserves reading order).
    Medium semi-random: decides constructs randomly but takes consecutive sentences for multi-sentence constructs.
    """
    idgen = IdGen(prefix=f"v{ch_idx}_")
    lines = []
    lines.append(build_header(ch_idx))
    lines.append("    _buffer = []\n")

    i = 0
    n = len(sent_list)

    text_ratio = max(0.05, min(0.95, float(text_ratio)))
    fake_factor = (1.0 - text_ratio) / max(text_ratio, 1e-9)

    def calc_fake_ops(text_lines_count: int) -> int:
        base = int(fake_factor * max(0, text_lines_count))
        jitter = rng.randint(0, max(1, int(fake_factor)))
        return max(0, base + jitter)

    def emit_fake_filler(fake_ops_count: int, indent: int = 4) -> str:
        indent_s = " " * indent
        out = []
        sneak_gap = max(3, int(12 * text_ratio))
        for _i in range(fake_ops_count):
            if (_i % sneak_gap == 0) and (i < n):
                snt = take_next(1)
                if snt:
                    r = rng.random()
                    if r < 0.4:
                        out.append(f"{indent_s}# {snt[0]}")
                    elif r < 0.8:
                        var = idgen.next()
                        out.append(f"{indent_s}{var} = {escape_str(snt[0])}")
                    else:
                        out.append(f"{indent_s}print(f{escape_str(snt[0])})")
            choice = rng.randint(0, 11)
            if choice == 0:
                out.append(f"{indent_s}_f{rng.randint(1,999)} = {rng.randint(0, 100)}")
            elif choice == 1:
                out.append(f"{indent_s}_n{rng.randint(1,999)} = sum([j for j in range({rng.randint(5,15)})])")
            elif choice == 2:
                out.append(f"{indent_s}def _fn_{rng.randint(1,999)}(x, y, z=0):")
                out.append(f"{indent_s}    return (x * y) - z + (x ** 2) / (y + 1)")
            elif choice == 3:
                out.append(f"{indent_s}[k for k in range({rng.randint(3,7)}) if k % 2 == 0]")
            elif choice == 4:
                out.append(f"{indent_s}expEU = [math.exp(beta * (u - maxEU)) for u in EU]")
            elif choice == 5:
                out.append(f"{indent_s}moves_count_decay[:] = [gamma * c for c in moves_count_decay]")
            elif choice == 6:
                out.append(f"{indent_s}expected_utility_recent = [empirical_prob_recent[(j - 1) % 3] for j in range(3)]")
            elif choice == 7:
                out.append(f"{indent_s}expected_utility = [expected_utility_all_time[ii] + b * expected_utility_recent[ii] for ii in range(3)]")
            elif choice == 8:
                out.append(f"{indent_s}_score = (alpha * beta) + gamma - (delta / max(1, n)) + (tau ** 2) - theta * omega")
            else:
                out.append(f"{indent_s}a, b, c = (x + 1, y * 2 - z, maxEU - minEU)")
        out.append("")
        return "\n".join(out)

    def take_next(k=1):
        nonlocal i
        taken = []
        for _ in range(k):
            if i >= n:
                break
            taken.append(sent_list[i])
            i += 1
        return taken

    # initial small header: a few comments then strings
    first_comments = rng.randint(1, 3)
    first_strings = rng.randint(1, 3)

    comments = take_next(first_comments)
    if comments:
        lines.append(emit_comment_lines(comments, indent=4))
        lines.append(emit_fake_filler(calc_fake_ops(len(comments)), indent=4))

    strings = take_next(first_strings)
    if strings:
        lines.append(emit_string_vars(strings, idgen, indent=4))
        # realistic append to buffer (use current numeric ids from idgen)
        # Since idgen.counter advanced by len(strings), we reconstruct names in order
        start_idx = idgen.counter - len(strings) + 1
        for offset in range(len(strings)):
            var_name = f"{idgen.prefix}{start_idx + offset:03d}"
            lines.append(f"    _buffer.append({var_name})")
        lines.append("")
        lines.append(emit_fake_filler(calc_fake_ops(len(strings)), indent=4))

    # chunked mix until consumed
    while i < n:
        choice = rng.choices(
            population=['comments','strings','prints','ifelse','forloop','match','try','class'],
            weights=[10, 12, 12, 20, 20, 10, 10, 15],
            k=1
        )[0]

        if choice == 'comments':
            cnt = rng.randint(1, 3)
            cs = take_next(cnt)
            if cs:
                lines.append(emit_comment_lines(cs, indent=4))
                lines.append(emit_fake_filler(calc_fake_ops(len(cs)), indent=4))

        elif choice == 'strings':
            cnt = rng.randint(1, 3)
            ss = take_next(cnt)
            if ss:
                lines.append(emit_string_vars(ss, idgen, indent=4))
                lines.append(emit_fake_filler(calc_fake_ops(len(ss)), indent=4))

        elif choice == 'prints':
            cnt = rng.randint(1, 3)
            ps = take_next(cnt)
            if ps:
                lines.append(emit_print_lines(ps, indent=4))
                lines.append(emit_fake_filler(calc_fake_ops(len(ps)), indent=4))

        elif choice == 'ifelse':
            c_comment = take_next(rng.randint(0,1))
            true_cnt = rng.randint(1, 3)
            true_sents = take_next(true_cnt)
            false_cnt = rng.randint(0, 2)
            false_sents = take_next(false_cnt)
            if c_comment or true_sents or false_sents:
                lines.append(emit_if_else(c_comment, true_sents, false_sents, idgen, indent=4))
                tlc = len(c_comment) + len(true_sents) + len(false_sents)
                lines.append(emit_fake_filler(calc_fake_ops(tlc), indent=4))

        elif choice == 'forloop':
            pre_comments = take_next(rng.randint(0,1))
            loop_count = rng.randint(1, 5)  # up to 5 in loop
            loop_sents = take_next(loop_count)
            fake_ops = rng.randint(0, 2)
            if pre_comments or loop_sents:
                lines.append(emit_for_loop(pre_comments, loop_sents, fake_ops, idgen, indent=4))
                tlc = len(pre_comments) + len(loop_sents)
                lines.append(emit_fake_filler(calc_fake_ops(tlc), indent=4))

        elif choice == 'match':
            ncases = rng.randint(3, 6)
            cases = []
            for ci in range(ncases):
                if i >= n:
                    break
                case_count = rng.randint(1, 2)
                case_sents = take_next(case_count)
                if case_sents:
                    cases.append((f"case_{ci}", case_sents))
            if cases:
                lines.append(emit_match_case(cases, idgen, indent=4))
                tlc = sum((1 if len(s) >= 1 else 0) + (1 if len(s) >= 2 else 0) for _, s in cases)
                lines.append(emit_fake_filler(calc_fake_ops(tlc), indent=4))

        elif choice == 'try':
            try_cnt = rng.randint(1, 3)
            exc_cnt = rng.randint(0, 2)
            try_sents = take_next(try_cnt)
            except_sents = take_next(exc_cnt)
            if try_sents:
                lines.append(emit_try_except(try_sents, except_sents, idgen, indent=4))
                tlc = len(try_sents) + len(except_sents)
                lines.append(emit_fake_filler(calc_fake_ops(tlc), indent=4))

        elif choice == 'class':
            class_cnt = rng.randint(1, 3)
            class_sents = take_next(class_cnt)
            if class_sents:
                cname = f"Node_{rng.randint(1,999):03d}"
                lines.append(emit_small_class(cname, class_sents, idgen, indent=4))
                lines.append(emit_fake_filler(calc_fake_ops(len(class_sents)), indent=4))

    # final decorative loop
    lines.append("    for _ in _buffer:")
    lines.append("        # reading pipeline (noop)")
    lines.append("        pass")
    lines.append("")
    return "\n".join(lines)

# -----------------------
# Orchestration
# -----------------------

def compile_book_to_code_fixed(epub_path: str, out_path: str, seed: int = None, text_ratio: float = 0.5):
    chapters = extract_chapters_from_epub(epub_path)
    if not chapters:
        raise RuntimeError("No content found in EPUB.")

    all_chapter_sentences = []
    for ch_text in chapters:
        paras = [p.strip() for p in ch_text.split("\n\n") if p.strip()]
        ch_sents = []
        for p in paras:
            sents = split_into_sentences(p)
            if sents:
                ch_sents.extend(sents)
        all_chapter_sentences.append(ch_sents)

    out_lines = []
    out_lines.append("# Auto-generated 'codebook' from EPUB")
    out_lines.append("# Generated by epub2code.py")
    out_lines.append("")
    out_lines.append("import math")
    out_lines.append("")
    out_lines.append("alpha = 1.0")
    out_lines.append("beta = 1.0")
    out_lines.append("gamma = 0.95")
    out_lines.append("delta = 0.01")
    out_lines.append("tau = 2.0")
    out_lines.append("theta = 0.1")
    out_lines.append("omega = 0.2")
    out_lines.append("maxEU = 1.0")
    out_lines.append("minEU = 0.0")
    out_lines.append("EU = [0.2, 0.5, 0.8]")
    out_lines.append("expected_utility_all_time = [0.0, 0.0, 0.0]")
    out_lines.append("empirical_prob_recent = [0.33, 0.33, 0.34]")
    out_lines.append("moves_count_decay = [0, 0, 0]")
    out_lines.append("counts = {'a': 1, 'b': 2, 'c': 3}")
    out_lines.append("total = sum(counts.values())")
    out_lines.append("step = 0")
    out_lines.append("user_id = 'user-000'")
    out_lines.append("state = 'idle'")
    out_lines.append("moves = [0, 1, 2, 1, 0]")
    out_lines.append("x, y, z = (1.0, 2.0, 0.5)")
    out_lines.append("b = 0.5")
    out_lines.append("n = 10")
    out_lines.append("")
    out_lines.append("class BookEngine:")
    out_lines.append("    \"\"\"Auto-generated book engine. Open in VS Code for fake-code reading experience.\"\"\"")
    out_lines.append("")

    global_rng = random.Random(seed)

    for i, ch_sent_list in enumerate(all_chapter_sentences, start=1):
        rng = random.Random(global_rng.randint(0, 10**9))
        # DO NOT shuffle: keep original reading order
        pool = list(ch_sent_list)
        chapter_code = build_chapter_function_sequential(pool, i, rng, text_ratio=text_ratio)
        # indent function inside class
        indented = []
        for line in chapter_code.splitlines():
            if line.strip() == "":
                indented.append("")
            else:
                indented.append("    " + line)
        out_lines.extend(indented)

    out_lines.append("")
    out_lines.append("if __name__ == '__main__':")
    out_lines.append("    # Nothing to run: open this file in an editor to read the book.")
    out_lines.append("    engine = BookEngine()")
    out_lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))

    print(f"[+] Wrote codebook to: {out_path}")

# -----------------------
# CLI
# -----------------------

def main():
    parser = argparse.ArgumentParser(description="Convert an EPUB to a fake-code .py file for reading in an editor.")
    parser.add_argument("epub", help="Path to input .epub file")
    parser.add_argument("-o", "--output", help="Output .py file path", default="book_as_code.py")
    parser.add_argument("--seed", help="Optional random seed (int) for reproducible layout", type=int, default=None)
    parser.add_argument("--ratio", help="Text-to-code ratio (0.05-0.95)", type=float, default=0.6)
    args = parser.parse_args()

    epub_path = args.epub
    out_path = args.output
    ratio = max(0.05, min(0.95, float(args.ratio)))
    if not os.path.isfile(epub_path):
        print("ERROR: epub file not found:", epub_path)
        return

    compile_book_to_code_fixed(epub_path, out_path, seed=args.seed, text_ratio=ratio)

if __name__ == "__main__":
    main()
