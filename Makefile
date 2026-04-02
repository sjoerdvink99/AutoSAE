.PHONY: ui server dev install

MODEL ?= meta-llama/Llama-3.1-8B-Instruct
PORT  ?= 8000

install:
	uv sync
	cd packages/ui && bun install

ui:
	cd packages/ui && bun dev

server:
	uv run autosae-server --model-id $(MODEL) --load-in-4bit --port $(PORT)

dev:
	@echo "Starting server and UI in parallel..."
	$(MAKE) server & $(MAKE) ui
