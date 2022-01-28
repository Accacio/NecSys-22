OBJ=article

all: $(OBJ).pdf

$(OBJ).pdf: $(OBJ).tex $(OBJ).bbl
	@echo PDF
	@pdflatex -interaction errorstopmode -synctex=1 $(OBJ).tex >/dev/null
	@pdflatex -interaction errorstopmode -synctex=1 $(OBJ).tex >/dev/null

$(OBJ).aux.1: $(OBJ).tex
	@echo CREATING AUX FILE
	@pdflatex  -interaction nonstopmode $(OBJ).tex >/dev/null
	@cp $(OBJ).aux $(OBJ).aux.1

$(OBJ).cites: $(OBJ).aux.1
	@echo VERIFYING CITATIONS
	@grep citation $(OBJ).aux.1 | sort > $@.1
	@touch $@.1
	@if [ ! -f "$@" ]; then touch $@; fi
	@if [ "`diff -N $@ $@.1`" != "" ];\
	then \
	echo " CITATIONS CHANGED";\
	cp $@.1 $@ ;\
	fi;
	@rm -f $@.1;
	@if [ -f "$@" ]; then touch $< -r $@; fi
	@if [ -f "$@" ]; then touch $(OBJ).tex -r $@; fi

$(OBJ).bbl: $(OBJ).cites bibliography.bib
	@echo BIBTEX
	@if [ "`diff -N $(OBJ).cites ""`" != "" ];\
	then bibtex $(OBJ) >/dev/null;\
	if [ "`grep bibitem $(OBJ).bbl`" = "" ];\
	then echo NO CITATION FOUND;\
	echo " " > $(OBJ).bbl;\
	fi;\
	else echo " " > $(OBJ).bbl;\
	fi


view: $(OBJ).pdf
	zathura $(OBJ).pdf&

clean:
	@rm -rf  $(OBJ).aux.1 $(OBJ).aux $(OBJ).out $(OBJ).log
	@rm -rf  $(OBJ).synctex.gz $(OBJ).upa $(OBJ).upb
	@rm -rf  $(OBJ).ist $(OBJ).acn $(OBJ).glsdefs $(OBJ).glo
	@rm -rf  $(OBJ).cites $(OBJ).bbl $(OBJ).blg $(OBJ).bcf

clean-all: clean
	@rm -rf $(OBJ).pdf
