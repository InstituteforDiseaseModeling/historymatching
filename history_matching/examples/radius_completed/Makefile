SHELL = /bin/bash

clean:
	for f in iter0 iter1 iter2 iter3; do rm -r $$f/Cuts; rm -r $$f/Data*; rm -r $$f/Plots; rm $$f/Candidates*; rm $$f/*.log; done

example:
	for f in iter0 iter1 iter2 iter3; do pushd $$f; python3 generate_samples_and_results.py; python3 bhm.py; python3 cut.py; python3 plot_implausibility.py; python3 plot_samples.py; popd; done

start:
	python3 iter0/generate_samples_and_results.py && python3 iter0/bhm.py --no-force-optimize-glm && python3 iter0/cut.py
	python3 iter0/plot_samples.py
	python3 iter0/plot_implausibility.py

next:
	python3 iter1/generate_samples_and_results.py && python3 iter1/bhm.py && python3 iter1/cut.py
	python3 iter1/plot_samples.py
	python3 iter1/plot_implausibility.py

# ==================

iter0/Candidates_for_iter1.csv:
	python3 iter0/generate_samples_and_results.py && python3 iter0/bhm.py --no-force-optimize-glm && python3 iter0/cut.py
	python3 iter0/plot_samples.py
	python3 iter0/plot_implausibility.py

iter0: iter0/Candidates_for_iter1.csv

iter1/Candidates_for_iter2.csv: iter0/Candidates_for_iter1.csv
	python3 iter1/generate_samples_and_results.py && python3 iter1/bhm.py && python3 iter1/cut.py
	python3 iter1/plot_samples.py
	python3 iter1/plot_implausibility.py

iter1: iter1/Candidates_for_iter2.csv

iter2/Candidates_for_iter3.csv: iter1/Candidates_for_iter2.csv
	python3 iter2/generate_samples_and_results.py && python3 iter2/bhm.py && python3 iter2/cut.py
	python3 iter2/plot_samples.py
	python3 iter2/plot_implausibility.py

iter2: iter2/Candidates_for_iter3.csv

iter3/Candidates_for_iter4.csv: iter2/Candidates_for_iter3.csv
	python3 iter3/generate_samples_and_results.py && python3 iter3/bhm.py --no-force-optimize-glm && python3 iter3/cut.py
	python3 iter3/plot_samples.py
	python3 iter3/plot_implausibility.py

iter3: iter3/Candidates_for_iter4.csv

all: iter3/Candidates_for_iter4.csv
