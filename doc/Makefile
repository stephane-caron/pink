# Makefile for Sphinx documentation
#
# Copyright (c) 2022 Stéphane Caron.
#
# This file is part of Pink.
#
# Pink is free software: you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# Pink is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Pink. If not, see <http://www.gnu.org/licenses/>.

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

upload: clean html latexpdf
	cp $(BUILDDIR)/latex/$(PDFFILE) $(BUILDDIR)/html/$(PDFFILE)
	rsync -auvz --delete-after $(BUILDDIR)/html/ $(ONLINEDIR)/
