# Makefile for Sphinx documentation
#
# Copyright 2022 Stéphane Caron

# You can also set these variables from the command line.
OUTPUTDIR = $(abspath ../_build)

.PHONY: build
build:  ## build the documentation
	(cd .. && sphinx-build $(CURDIR) $(OUTPUTDIR) -W)

.PHONY: clean
clean:  ## remove all output files
	rm -rf $(OUTPUTDIR)

.PHONY: open
open:  ## open the locally built documentation in Firefox
	firefox $(OUTPUTDIR)/index.html
