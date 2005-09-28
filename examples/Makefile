# D. Brugger, September 2005
# Makefile
# $Id$

SOURCES = example1.ml
RESULT = example1
INCDIRS = $(shell ocamlfind -query lacaml) \
	$(shell ocamlfind -query libsvm)		
LIBS = bigarray lacaml libsvm

-include OCamlMakefile
