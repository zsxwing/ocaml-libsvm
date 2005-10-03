# D. Brugger, september 2005
# Makefile
# $Id$

SOURCES = libsvm_c.c libsvm.mli libsvm.ml
RESULT = libsvm
LIBINSTALL_FILES = $(wildcard *.mli *.cmi *.cma *.cmxa *.a *.so)
INCDIRS = $(shell ocamlc -v | tail -n 1 | cut -f 4 -d " ")/site-packages/lacaml
LIBDIRS = -L /usr/lib/libsvm
LIBS = lacaml
CLIBS = svm
CFLAGS = -Wall -ansi

ANNOTATE = yes

all: byte

byte: byte-code-library

opt: native-code-library

install: libinstall

uninstall: libuninstall

update: uninstall install

-include OCamlMakefile

