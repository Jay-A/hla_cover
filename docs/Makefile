# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line or environment
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR     = build

# Default target is help
help:
	@echo "Sphinx documentation makefile"
	@echo "Usage: make [target]"
	@echo "Available targets:"
	@echo "  html     - Build HTML documentation"
	@echo "  latex    - Build LaTeX documentation"
	@echo "  clean    - Remove built documentation"
	@echo "  help     - Show this help message"
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

.PHONY: help html latex clean

# Catch-all target to route all other targets to sphinx-build -M <target>
%: Makefile
	@echo "Building documentation target: $@"
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
