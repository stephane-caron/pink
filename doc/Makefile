# Makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = src
BUILDDIR      = build
ONLINEDIR     = scaron.info:public_html/doc/pink
PDFFILE       = pink.pdf

.PHONY: help Makefile upload

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

upload: html latexpdf
	cp $(BUILDDIR)/latex/$(PDFFILE) $(BUILDDIR)/html/$(PDFFILE)
	rsync -auvz --delete-after $(BUILDDIR)/html/ $(ONLINEDIR)/
