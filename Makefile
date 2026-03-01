all: data/cv.pdf data/publist.pdf data/bg.jpg index.html cv.html code.html

data/cv.pdf: tex/cv.tex tex/employment.tex tex/education.tex tex/publications.tex tex/talks.tex tex/teaching.tex tex/advising.tex tex/service.tex tex/honors.tex tex/funding.tex tex/res.cls scripts/make_cv
	./scripts/make_cv ''

data/publist.pdf: tex/publist.tex tex/publications.tex scripts/make_publist
	./scripts/make_publist ''

src/employment.src.html: tex/employment.tex scripts/make_employment
	./scripts/make_employment

src/education.src.html: tex/education.tex scripts/make_education
	./scripts/make_education

src/publications.src.html src/publications_split.src.html: tex/publications.tex scripts/make_publications
	./scripts/make_publications

src/talks.src.html: tex/talks.tex scripts/make_talks
	./scripts/make_talks

src/teaching.src.html: tex/teaching.tex scripts/make_teaching
	./scripts/make_teaching

src/honors.src.html: tex/honors.tex scripts/make_honors
	./scripts/make_honors

src/service.src.html: tex/service.tex scripts/make_service
	./scripts/make_service

src/funding.src.html: tex/funding.tex scripts/make_funding
	./scripts/make_funding

src/advising.src.html: tex/advising.tex scripts/make_advising
	./scripts/make_advising

index.html: src/template.html src/index.src.html scripts/make_html
	./scripts/make_html 'index.html'

cv.html: src/template.html src/cv.src.html src/employment.src.html src/education.src.html src/publications_split.src.html src/talks.src.html src/teaching.src.html src/honors.src.html src/service.src.html src/advising.src.html src/funding.src.html scripts/make_html
	./scripts/make_html 'cv.html'

code.html: src/template.html src/code.src.html scripts/make_html
	./scripts/make_html 'code.html'

data/bg.jpg: data/TNG300_projected_DM.npy data/TNG300_projected_PE.npy scripts/make_background
	./scripts/make_background

.PHONY: clean
clean:
	rm -f tmp/*
	rm ./index.html ./cv.html ./code.html
	rm src/publications*.src.html
	rm src/talks.src.html
	rm data/cv.pdf
	rm data/bg.jpg
