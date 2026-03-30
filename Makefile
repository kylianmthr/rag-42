SRC=student

all: ${NAME}

install:
	uv sync

run: ${NAME}
	uv run python -m ${SRC}

debug:
	uv run python -m pdb -m ${SRC}

clean:
	find . -iname "__pycache__" -type d -exec rm -rf "{}" +
	find . -iname ".mypy_cache" -type d -exec rm -rf "{}" +

lint:
	uv run flake8 ${SRC}
	uv run mypy ${SRC} --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs --exclude "src/llm_sdk/" --follow-imports=silent

lint-strict:
	uv run flake8 ${SRC}
	uv run mypy ${SRC} --strict --exclude "src/llm_sdk/" --follow-imports=silent

#TODO: A SUPPRIMER
test:
	uv run pytest -vv

.PHONY: install run debug clean lint lint-strict test
