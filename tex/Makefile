
.PHONY: all clean force-build

# define default targets for `make
all: presentation.pdf

clean:
	rm -f *.aux
	rm -f *.bbl
	rm -f *.bcf
	rm -f *.blg
	rm -f *.fdb_latexmk
	rm -f *.fls
	rm -f *.log
	rm -f *.nav
	rm -f *.out
	rm -f *.run.xml
	rm -f *.snm
	rm -f *.synctex.gz
	rm -f *.toc
	rm -f *.xml

# this target is never made, listing it as a dependency forces the make tasks
# below to be always executed.
force-build:

presentation.pdf: force-build
	pdflatex presentation
	biber presentation
	pdflatex presentation
	pdflatex presentation
