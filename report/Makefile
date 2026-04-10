# Makefile for report/report.tex
# Requires cam-thesis.cls — copy it first with: make setup

TEMPLATE_DIR = ../Cambridge_University_Computer_Lab_Thesis_Dissertation_Template
SHIELDS_DIR  = $(TEMPLATE_DIR)/CollegeShields

.PHONY: all setup clean

all: report.pdf

setup:
	cp $(TEMPLATE_DIR)/cam-thesis.cls .
	@if [ -d $(SHIELDS_DIR) ]; then \
	    cp -r $(SHIELDS_DIR) . ; \
	fi

report.pdf: report.tex report.bib
	pdflatex report
	bibtex report
	pdflatex report
	pdflatex report

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.lol \
	      *.glg *.glo *.gls *.ist *.idx *.ilg *.ind
