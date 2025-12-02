.PHONY: start install clean

# Activate virtual environment and run the application
start:
	@echo "Starting 2D to 3D converter..."
	@. .venv/bin/activate && python src/main.py

# Install dependencies in virtual environment
install:
	@echo "Installing dependencies..."
	@python3 -m venv .venv
	@. .venv/bin/activate && pip install -r requirements.txt
	@echo "Dependencies installed successfully!"

# Clean up virtual environment
clean:
	@echo "Removing virtual environment..."
	@rm -rf .venv
	@echo "Virtual environment removed!"
