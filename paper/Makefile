# Makefile — SRFM Research Paper
# Usage:
#   make pdf          Build the paper PDF (full LaTeX compile + bibtex)
#   make figures      Generate all Python figures
#   make clean        Remove build artifacts
#   make distclean    Remove everything except source
#   make watch        Continuous rebuild on file change (requires entr)
#   make wordcount    Count words in the paper

# ── Configuration ─────────────────────────────────────────────────────────────
LATEX     := pdflatex
BIBTEX    := bibtex
PYTHON    := python3
MAIN      := main
OUT_DIR   := build

LATEX_FLAGS := -interaction=nonstopmode -output-directory=$(OUT_DIR)

# Source files (triggers rebuild)
TEX_SOURCES := main.tex \
               sections/01_abstract.tex \
               sections/02_introduction.tex \
               sections/03_related_work.tex \
               sections/04_framework.tex \
               sections/05_implementation.tex \
               sections/06_empirical.tex \
               sections/07_open_questions.tex \
               sections/08_conclusion.tex \
               srfm.sty \
               bibliography.bib

FIGURE_SCRIPTS := $(wildcard figures/gen_*.py)
FIGURE_PDFS    := $(patsubst figures/gen_%.py,figures/%.pdf,$(FIGURE_SCRIPTS))

# ── Default target ────────────────────────────────────────────────────────────
.PHONY: all
all: pdf

# ── PDF compilation ────────────────────────────────────────────────────────────
.PHONY: pdf
pdf: $(OUT_DIR)/$(MAIN).pdf

$(OUT_DIR):
	mkdir -p $(OUT_DIR)

$(OUT_DIR)/$(MAIN).pdf: $(TEX_SOURCES) $(FIGURE_PDFS) | $(OUT_DIR)
	@echo "[srfm] Pass 1: pdflatex..."
	$(LATEX) $(LATEX_FLAGS) $(MAIN).tex
	@echo "[srfm] bibtex..."
	cd $(OUT_DIR) && $(BIBTEX) $(MAIN)
	@echo "[srfm] Pass 2: pdflatex..."
	$(LATEX) $(LATEX_FLAGS) $(MAIN).tex
	@echo "[srfm] Pass 3: pdflatex (resolving cross-references)..."
	$(LATEX) $(LATEX_FLAGS) $(MAIN).tex
	@echo "[srfm] Built: $(OUT_DIR)/$(MAIN).pdf"
	@cp $(OUT_DIR)/$(MAIN).pdf ./$(MAIN).pdf
	@echo "[srfm] Copied to: ./$(MAIN).pdf"

# ── Figures ────────────────────────────────────────────────────────────────────
.PHONY: figures
figures: $(FIGURE_PDFS)

figures/%.pdf: figures/gen_%.py
	@echo "[srfm] Generating figure: $@"
	$(PYTHON) $<

# Run all figure scripts at once (faster)
.PHONY: figures-all
figures-all:
	@echo "[srfm] Generating all figures..."
	$(PYTHON) figures/gen_all.py

# ── Quick check (nonstop mode, single pass) ────────────────────────────────────
.PHONY: check
check: | $(OUT_DIR)
	$(LATEX) $(LATEX_FLAGS) $(MAIN).tex
	@grep -i "error\|undefined\|warning" $(OUT_DIR)/$(MAIN).log | head -30 || true

# ── Word count ────────────────────────────────────────────────────────────────
.PHONY: wordcount
wordcount:
	@if command -v texcount >/dev/null 2>&1; then \
	    texcount -sum -1 sections/*.tex; \
	else \
	    echo "[srfm] texcount not found; using wc -w as fallback"; \
	    cat sections/*.tex | grep -v '^%' | wc -w; \
	fi

# ── Continuous rebuild ─────────────────────────────────────────────────────────
.PHONY: watch
watch:
	@echo "[srfm] Watching for changes (requires 'entr')..."
	ls $(TEX_SOURCES) | entr make pdf

# ── Clean ─────────────────────────────────────────────────────────────────────
.PHONY: clean
clean:
	rm -rf $(OUT_DIR)
	rm -f $(MAIN).pdf

.PHONY: distclean
distclean: clean
	rm -f figures/*.pdf
	rm -f *~ sections/*~

# ── arXiv submission package ──────────────────────────────────────────────────
.PHONY: arxiv
arxiv: pdf
	@echo "[srfm] Building arXiv submission tarball..."
	mkdir -p arxiv_submit
	cp main.tex arxiv_submit/
	cp srfm.sty arxiv_submit/
	cp bibliography.bib arxiv_submit/
	cp -r sections arxiv_submit/
	cp -r figures/*.pdf arxiv_submit/figures/ 2>/dev/null || mkdir -p arxiv_submit/figures
	tar -czf srfm_arxiv_$(shell date +%Y%m%d).tar.gz -C arxiv_submit .
	rm -rf arxiv_submit
	@echo "[srfm] Tarball: srfm_arxiv_$(shell date +%Y%m%d).tar.gz"

.PHONY: help
help:
	@echo "SRFM Paper Build System"
	@echo ""
	@echo "Targets:"
	@echo "  make pdf          Full PDF build (3-pass LaTeX + BibTeX)"
	@echo "  make figures      Generate all Python figures"
	@echo "  make figures-all  Run gen_all.py (faster parallel generation)"
	@echo "  make check        Single-pass compile for syntax checking"
	@echo "  make wordcount    Count words in section files"
	@echo "  make watch        Continuous rebuild on file change"
	@echo "  make arxiv        Build arXiv submission tarball"
	@echo "  make clean        Remove build artifacts"
	@echo "  make distclean    Remove build artifacts + figures"
