build_torchffi:
	(cd torchffi/ && ./build.sh)

setup_venv:
	python3 -m venv ../venv
	(. ../venv/bin/activate && pip install -r test_gen/requirements.txt)