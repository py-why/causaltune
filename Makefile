package_name = auto_causality
coverage_target = 70
max_line_length = 88

venv_name = venv
venv_activate_path := ./$(venv_name)/bin/activate
cov_args := --cov $(package_name) --cov-fail-under=$(coverage_target) --cov-report=term-missing
not_slow = -m "not slow"

.PHONY: clean venv update lint test slowtest cov slowcov format checkformat

clean:
	rm -rf ./$(venv_name)

venv:
	python3 -m venv $(venv_name) ;\
	. $(venv_activate_path) ;\
	pip install --upgrade pip setuptools wheel ;\
	pip install --upgrade -r requirements-dev.txt ;\
	pip install --upgrade -r requirements.txt

update:
	. $(venv_activate_path) ;\
	pip install --upgrade -r requirements-dev.txt ;\
	pip install --upgrade -r requirements.txt

lint:
	. $(venv_activate_path) ;\
	flake8 --max-line-length=$(max_line_length)

test:
	. $(venv_activate_path) ;\
	py.test $(not_slow) --disable-warnings

slowtest:
	. $(venv_activate_path) ;\
	py.test

cov:
	. $(venv_activate_path) ;\
	py.test $(cov_args) $(not_slow)

slowcov:
	. $(venv_activate_path) ;\
	py.test $(cov_args)

format:
	. $(venv_activate_path) ;\
	black $(package_name)/ --skip-string-normalization
	black tests/ --skip-string-normalization

checkformat:
	. $(venv_activate_path) ;\
	black $(package_name)/ --skip-string-normalization --check ;\
	black tests/ --skip-string-normalization --check
