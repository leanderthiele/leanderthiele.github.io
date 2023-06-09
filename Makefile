all: index.html cv.html code.html data/cv.pdf data/bg.jpg

data/cv.pdf: tex/cv.tex tex/publications.tex tex/res.cls make_cv
	./make_cv

src/publications.src.html src/publications_split.src.html: tex/publications.tex make_publications
	./make_publications

index.html: src/template.html src/index.src.html make_html
	./make_html 'index.html'

cv.html: src/template.html src/cv.src.html src/publications_split.src.html make_html
	./make_html 'cv.html'

code.html: src/template.html src/code.src.html make_html
	./make_html 'code.html'

data/bg.jpg: data/TNG300_projected_DM.npy data/TNG300_projected_PE.npy make_background
	./make_background

.PHONY: clean
clean:
	rm tmp/*
	rm docs/index.html docs/cv.html docs/code.html
	rm src/publications*.src.html
	rm data/cv.pdf
	rm data/bg.jpg
