# $Id$
# Adapted from ocaml-glpk

LIBNAME = ocaml-libsvm
DISTFILES = CHANGES COPYING Makefile README VERSION \
            src/Makefile src/OCamlMakefile src/*.ml src/*.mli src/*.c \
	    doc/html
INCDIRS = $(shell ocamlc -v | tail -n 1 | cut -f 4 -d " ")/site-packages/lacaml

all: byte

byte opt clean install uninstall:
	make -C src $@

distclean: clean
	rm -rf doc

doc: byte
	mkdir -p doc/html
	ocamldoc -html -m A -I $(INCDIRS) -I src -d doc/html src/libsvm.mli

dist: doc
	VERSION="$(shell cat VERSION)"; \
	mkdir $(LIBNAME)-$$VERSION; \
	cp -r --parents $(DISTFILES) $(LIBNAME)-$$VERSION; \
	tar cvzf $(LIBNAME)-$$VERSION.tar.gz $(LIBNAME)-$$VERSION; \
	rm -rf $(LIBNAME)-$$VERSION

.PHONY: doc
