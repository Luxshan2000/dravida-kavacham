# Define the virtual environment folder
VENV = .venv

.PHONY: setup
setup: venv requirements

venv:
	@echo "Creating virtual environment in $(VENV)..."
	python -m venv $(VENV)

requirements: venv
	@echo "Installing dependencies..."
	$(VENV)/Scripts/pip install -r requirements.txt

clean:
	@echo "Cleaning up the virtual environment..."
	@if exist $(VENV) rd /s /q $(VENV)
	@if exist __pycache__ rd /s /q __pycache__
	@if exist output rd /s /q output
