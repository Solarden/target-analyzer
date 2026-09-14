.DEFAULT_GOAL := help
.PHONY: help css css-watch tailwind-cli

# Tailwind standalone CLI (no Node needed). The binary is a dev tool — fetched on
# demand and gitignored; the built CSS is committed and served offline through
# StaticFiles, exactly like the vendored chart.min.js. See tailwind.css.
TAILWIND_VERSION := v4.3.0
TAILWIND_BIN := tools/tailwindcss
TAILWIND_INPUT := tailwind.css
TAILWIND_OUTPUT := server/target_analyzer/static/app.css

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

tailwind-cli: ## Fetch the standalone Tailwind CLI if it isn't here yet
	@test -x $(TAILWIND_BIN) && echo "Tailwind CLI present" || ( \
		mkdir -p tools && \
		os=$$(uname -s) ; arch=$$(uname -m) ; \
		case "$$os-$$arch" in \
			Darwin-arm64) asset=tailwindcss-macos-arm64 ;; \
			Darwin-x86_64) asset=tailwindcss-macos-x64 ;; \
			Linux-aarch64|Linux-arm64) asset=tailwindcss-linux-arm64 ;; \
			Linux-x86_64) asset=tailwindcss-linux-x64 ;; \
			*) echo "unsupported platform $$os-$$arch" >&2 ; exit 1 ;; \
		esac ; \
		echo "downloading $$asset ($(TAILWIND_VERSION))" ; \
		curl -sL -o $(TAILWIND_BIN) \
			https://github.com/tailwindlabs/tailwindcss/releases/download/$(TAILWIND_VERSION)/$$asset && \
		chmod +x $(TAILWIND_BIN) )

css: tailwind-cli ## Build the dashboard CSS (minified) into the static dir
	$(TAILWIND_BIN) -i $(TAILWIND_INPUT) -o $(TAILWIND_OUTPUT) --minify
	@printf '\n' >> $(TAILWIND_OUTPUT)  # trailing newline, so end-of-file-fixer leaves it alone

css-watch: tailwind-cli ## Rebuild the CSS on template changes while developing
	$(TAILWIND_BIN) -i $(TAILWIND_INPUT) -o $(TAILWIND_OUTPUT) --watch
