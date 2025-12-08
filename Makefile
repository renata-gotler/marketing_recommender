install:
	@echo "Installing requirements..."
	pip install --upgrade pip &&\
		pip install -r requirements.txt

install-dev:
	@echo "Installing dev requirements..."
	pip install --upgrade pip &&\
		pip install -r requirements/requirements-dev.txt

fix:
	@echo "Fixing code..."
	ruff --fix

format:
	@echo "Formatting code..."
	ruff format .

test:
	@echo "Running tests..."
	pytest $(args)

update-branch:
	git config --global user.name $(USER_NAME)
	git config --global user.email $(USER_EMAIL)
	git commit -am "Improve formatting and clean code"
	git push --force origin HEAD:update