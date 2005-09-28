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

#INCLUDE="/usr/lib/ocaml"
#CC=gcc
#OCAMLC=ocamlc
#OCAMLMKLIB=ocamlmklib
#OCAMLFIND=ocamlfind
#OCAMLDOC=ocamldoc

#all : test.cma libsvm.cma

#doc : libsvm.mli
#	mkdir html
#	$(OCAMLFIND) $(OCAMLDOC) -d html -html $< -package lacaml

#test : test.ml
#	$(OCAMLC) -c $<

#libsvm.cmo : libsvm.ml libsvm.cmi
#	$(OCAMLFIND) $(OCAMLC) -c $< -I ../util matlab.cmo -package lacaml

#libsvm.cmi : libsvm.mli
#	$(OCAMLFIND) $(OCAMLC) -c $< -I ../util matlab.cmo -package lacaml

#mytest : mytest.c
#	$(CC) -c -I $(INCLUDE) $<

#libsvm_c : libsvm_c.c
#	$(CC) -c -I $(INCLUDE) $<

#libsvm.cma : libsvm_c libsvm.cmo libsvm.cmi
#	$(OCAMLMKLIB) -o libsvm libsvm.cmo libsvm_c.o -L"/usr/lib/libsvm" -lsvm -lm

#test.cma : mytest test
#	$(OCAMLMKLIB) -o test test.cmo mytest.o

#clean :
#	rm -rf html mytest.o *~ *.cmi *.cmo *.cma
