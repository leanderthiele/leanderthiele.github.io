all: index.html cv.html code.html data/cv.pdf

data/cv.pdf: tex/cv.tex tex/publications.tex tex/res.cls make_cv
	./make_cv

src/publications.src.html src/publications_split.src.html: tex/publications.tex make_publications
	./make_publications

index.html cv.html code.html: src/template.html src/index.src.html src/cv.src.html src/code.src.html src/publications_split.src.html make_html
	./make_html

.PHONY: clean
clean:
	rm tmp/*
	rm index.html cv.html code.html
	rm src/publications*.src.html
	rm data/cv.pdf
