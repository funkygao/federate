SHELL := /bin/bash
.SILENT:

FEDERATE := federate
INVENTORY_FILE := inventory.yaml
ENV_FILE := .env
INVENTORY_CMD := $(FEDERATE) inventory -i $(INVENTORY_FILE)
FUSION_PROJECTS := fusion-projects
TARGET_ENV ?= $$(cat $(ENV_FILE) 2>/dev/null)

help:
	awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make ENV=test \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-30s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

include .common.mk
include $(FUSION_PROJECTS)/.*.mk

##@ Submodule

select-env:
	if [ -z "$(ENV)" ]; then \
		PS3="切换到哪一个环境？输入对应的数字："; \
		select CHOSEN_ENV in $$($(INVENTORY_CMD) -f env-list); do \
			if [ -n "$${CHOSEN_ENV}" ]; then \
				echo "$${CHOSEN_ENV}" > $(ENV_FILE); \
				break; \
			fi; \
		done; \
		ENV=$$(cat $(ENV_FILE)); \
	else \
		echo "$${ENV}" > $(ENV_FILE); \
	fi; \
	echo "目标环境：$${ENV}"

current-env: ## Display current environment and its configuration
	$(INVENTORY_CMD) -f repo-info -e $(TARGET_ENV)

confirm-env:
	@echo "Current environment is: $(TARGET_ENV)"
	@read -p "Do you want to proceed with this environment? [y/N] " confirm; \
	if [ "$$confirm" != "y" ] && [ "$$confirm" != "Y" ]; then \
		echo "Operation cancelled."; \
		exit 1; \
	fi

sync-submodules:select-env restore-submodules ## Initialize all git-submodules and pull latest remote commits.
	ENV=$$(cat $(ENV_FILE)); \
	$(INVENTORY_CMD) -f sync-submodules -e $$ENV | while read cmd; do \
		echo "Executing: $$cmd"; \
		eval $$cmd; \
	done; \
	git submodule update --init --recursive --remote; \
	echo "🍺 All submodules synchronized for $$ENV environment."
	#git checkout -q .gitmodules

diff-submodules: ## Display local changes compared with their original state.
	git submodule foreach --recursive 'git diff --ignore-space-at-eol -w -b'

restore-submodules:
	for repo in $$($(INVENTORY_CMD) -f repo-list); do \
		if [ -d "$$repo" ]; then \
			echo "Checking out $$repo..."; \
			(cd $$repo && git checkout -q .) || echo "Failed to checkout $$repo"; \
		fi; \
	done

clean:
	rm -rf logs generated test.log dependency-tree.txt
	find . \( -name "prompt.txt" -o -name "out.txt" -o -name .DS_Store -o -name "*.log" \) -exec rm {} \;
	$(MAKE) restore-submodules

push-github:
	git subtree split --prefix=cmd/federate -b split-branch
	git push github split-branch:main
	git branch -D split-branch
