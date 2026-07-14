.PHONY: install run test clean

install:
	python -m pip install -r requirements.txt

run:
	python -m src.pipeline

test:
	python -m pytest -q

clean:
	rm -rf data artifacts reports .pytest_cache src/__pycache__ tests/__pycache__
