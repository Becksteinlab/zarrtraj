# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = $(shell echo $$SPHINXBUILD_ENV)
SPHINXPROJ    = zarrtraj
SOURCEDIR     = source
BUILDDIR      = _build

# Check if SPHINXBUILD is not set or is empty and throw an error
ifeq ($(SPHINXBUILD),)
$(error SPHINXBUILD_ENV environment variable is not set. Please define SPHINXBUILD_ENV with the path to the sphinx-build executable.)
endif

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)