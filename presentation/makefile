OBJ=presentation
all: $(OBJ).pdf

$(OBJ).pdf: $(OBJ).tex
	@echo PDF
	@pdflatex $(OBJ).tex

script: $(OBJ).tex
	@echo MAKING SCRIPT
	@cat $(OBJ).tex | egrep "^ *\\\script" | sed -E "s,\s*\\\script(<.*>|)\{(.*)\},\2,g" > script

clean:
	@rm -rf  $(OBJ).aux.1 $(OBJ).aux $(OBJ).out $(OBJ).log
	@rm -rf  $(OBJ).synctex.gz $(OBJ).upa $(OBJ).upb
	@rm -rf  $(OBJ).ist $(OBJ).acn $(OBJ).glsdefs $(OBJ).glo
	@rm -rf  $(OBJ).snm $(OBJ).nav $(OBJ).toc $(OBJ).vrb


clean-all: clean
	rm -rf $(OBJ).pdf
# end
