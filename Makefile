package_name = causaltune
max_line_length = 120

venv_name = venv
venv_activate_path := ./$(venv_name)/bin/activate
not_slow = -m "not slow"
# Run tests across all cores. --dist load distributes individual tests (incl.
# parametrized cases) freely across workers; loadscope would pin a whole class/
# module to one worker, serializing our parametrized end-to-end tests. The
# module-scoped `data` fixtures are cheap synthetic datasets, so rebuilding them
# per worker costs far less than the parallelism we gain.
parallel = -n auto --dist load
# Pin every numeric/BLAS/joblib backend to a single thread. Each CausalTune fit
# defaults to components_njobs=-1 and the learners spin up OpenMP/BLAS pools; run
# under xdist that means every worker grabs all cores and they thrash (xdist then
# gives ~no speedup). Capping to one thread per process makes parallelism come
# from xdist workers instead of nested oversubscription.
thread_caps = OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
	NUMEXPR_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1

.PHONY: venv test slowtest lint format checkformat

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
	$(thread_caps) py.test $(parallel) $(not_slow) --disable-warnings

slowtest:
	. $(venv_activate_path) ;\
	$(thread_caps) py.test $(parallel)

format:
	. $(venv_activate_path) ;\
	isort -rc .
	autoflake -r --in-place --remove-unused-variables .
	black $(package_name)/ --skip-string-normalization
	black tests/ --skip-string-normalization

checkformat:
	. $(venv_activate_path) ;\
	black $(package_name)/ --skip-string-normalization --check ;\
	black tests/ --skip-string-normalization --check
