.PHONY: setup test serve preview clean

setup:
	./setup.sh

test:
	python3 tests/test_bandit.py
	python3 tests/test_pool.py

preview:
	python3 -m fugu_swarm.run

serve:
	python3 -m fugu_swarm.run --serve

clean:
	rm -rf vendor artifacts __pycache__ fugu_swarm/__pycache__ tests/__pycache__ .pytest_cache
