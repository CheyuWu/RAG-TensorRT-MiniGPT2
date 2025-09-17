



install:
	pip3 install -r requirements.txt

lint:
	ruff check .
	black --check .
format:
	black .
	ruff check .

test:
	pytest . -s
