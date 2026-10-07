# Local mirror of the CI/CD pipeline:  make ci
PY ?= python

.PHONY: setup lint hygiene test data train eval gate package smoke ci app docker-build clean

setup:      ## install runtime + dev dependencies
	$(PY) -m pip install -r requirements-dev.txt
lint:       ## PEP 8 / pyflakes via ruff
	ruff check .
hygiene:    ## secrets, PII, forbidden files
	$(PY) scripts/check_repo_hygiene.py
test:
	$(PY) -m pytest
data:       ## validate training data (schema, ranges, labels)
	$(PY) scripts/validate_data.py
train:      ## train + cross-validate the scorer
	$(PY) scripts/train_model.py
eval:       ## extraction / Q&A / safety evaluation
	$(PY) scripts/evaluate.py
gate:       ## quality gates (fails the build on regression)
	$(PY) scripts/ci_quality_gate.py
package:    ## manifest + model card (requires passed gates)
	$(PY) scripts/package_model.py --require-gates
smoke:
	$(PY) scripts/smoke_test.py
ci: lint hygiene test data train eval gate package smoke   ## everything the CI server runs
	@echo "CI pipeline OK"
app:
	streamlit run app/streamlit_app.py
docker-build:
	docker build -t coverwise-ai:local .
clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__
