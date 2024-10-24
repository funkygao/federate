SHELL := /bin/bash
.SILENT:

FEDERATE_CMD ?= federate
INVENTORY_CMD := $(FEDERATE_CMD) inventory

ENV ?=

ENV_FILE := .env
TARGET_ENV ?= $$(cat $(ENV_FILE) 2>/dev/null)

##@ General Options

##   Use sci=1 with merge targets to Skip Components Installation.

help: 
	awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make ENV=test \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-30s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } /^##[^@]/ { printf "%s\n", substr($$0, 4) }' $(MAKEFILE_LIST)

##@ Submodule

select-env:
	if [ -z "$(ENV)" ]; then \
		PS3="切换到哪一个部署环境？输入对应的数字："; \
		select CHOSEN_ENV in $$($(INVENTORY_CMD) -f env-list); do \
			if [ -n "$${CHOSEN_ENV}" ]; then \
				echo "$${CHOSEN_ENV}" > $(ENV_FILE); \
				break; \
			else \
				echo "您输入的是 '$$REPLY'，请输入一个有效的数字。"; \
			fi; \
		done; \
		ENV=$$(cat $(ENV_FILE)); \
	else \
		echo "$${ENV}" > $(ENV_FILE); \
	fi; \
	echo "🍺 你选择的目标部署环境：$${ENV}"

current-env:
	$(INVENTORY_CMD) -f repo-info -e $(TARGET_ENV)

sync-submodules: ## Clean and synchronize submodules to latest remote commits.
	if [ -z "$(ENV)" ]; then \
		$(MAKE) select-env; \
		ENV=$$(cat $(ENV_FILE)); \
	fi; \
	echo "☕️ Restoring all submodules repository ..."; \
	$(MAKE) restore-submodules; \
	echo "☕️ Syncing all submodules repository for $$ENV environment"; \
	$(INVENTORY_CMD) -f sync-submodules -e $$ENV | while read cmd; do \
		echo "Executing: $$cmd"; \
		eval $$cmd; \
	done; \
	git submodule update --init --recursive --remote; \
	echo "🍺 Submodules synchronized for $$ENV environment."; \

diff-submodules: ## Display local changes compared with their original state.
	git submodule foreach --recursive 'git diff --ignore-space-at-eol -w -b'

restore-submodules:
	echo "☕️ Restoring submodules ..."
	for repo in $$($(INVENTORY_CMD) -f repo-list); do \
		if [ -d "$$repo" ]; then \
			echo "Checking out $$repo..."; \
			(cd $$repo && git checkout -q .) || echo "Failed to checkout $$repo"; \
		fi; \
	done

clean:restore-submodules
	rm -rf generated test.log

##@ Consolidate

consolidate: ## Generate the test environment target system.
	echo "☕️ Clearing up all generated target projects ..."
	$(MAKE) clean
	echo "☕️ Scaffolding target project fusion-starter ..."
	federate microservice fusion-start
	echo "☕️ Consolidating the target project ..."
	federate microservice consolidate --yes --yaml-conflict-cell-width 32 --silent=false
	if [ "$(sci)" != "1" ]; then \
		echo "☕️ Local installing the Instrumented Components ..."; \
		for repo in $$(federate components); do \
			profile=$$($(INVENTORY_CMD) -f maven-profile -r $$repo -e $$ENV); \
			modules=$$($(INVENTORY_CMD) -f maven-modules -r $$repo); \
			echo "Installing $$repo with profile:$$profile on modules:$$modules"; \
			(cd $$repo && mvn clean install -q -pl ":$$modules" -P"$$profile" -am -T8 -Dmaven.test.skip=true -Dfederate.packaging=true) || exit 1; \
		done; \
	else \
		echo "☕️ Skipped Components Installation"; \
	fi
	FUSION_STARTER_DIR={{.FusionStarter}} \
	echo "☕️ Local installing {{.FusionStarter}}.jar ..."; \
	$(MAKE) -C $$FUSION_STARTER_DIR install; \
	echo "☕️ Optimizing target project ..."; \
	federate microservice optimize --yes; \
	echo "🍺 Congrat, consolidated!"
