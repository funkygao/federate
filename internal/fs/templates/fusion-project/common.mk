# merge_project - Merge and consolidate a microservice project
#
# Usage:
#   $(call merge_project,<manifest_file>,<environment>[,<silent>])
#
# Parameters:
#   $(1) - Path to the manifest file (required)
#   $(2) - Environment (e.g., test, prod) (required)
#   $(3) - Silent mode for consolidation (optional, default: false)
#
# Options:
#   sci=1 - Skip Components Installation
#
# Example:
#   $(call merge_project,path/to/manifest.yaml,test,true)
define merge_project
	echo "☕️ Clearing up all generated target projects ..."
	$(MAKE) clean
	echo "☕️ Scaffolding target project fusion-starter ..."
	federate microservice fusion-start -i $(1)
	echo "☕️ Consolidating the target project ..."
	SILENT=$(if $(3),$(3),false); \
	federate microservice consolidate -i $(1) --yes --yaml-conflict-cell-width 32 --silent=$$SILENT
	if [ "$(sci)" != "1" ]; then \
		echo "☕️ Local installing the Instrumented Components ..."; \
		for repo in $$(federate components -i $(1)); do \
			profile=$$($(INVENTORY_CMD) -f maven-profile -r $$repo -e $(2)); \
			modules=$$($(INVENTORY_CMD) -f maven-modules -r $$repo); \
			echo "Installing $$repo with profile:$$profile on modules:$$modules"; \
			(cd $$repo && mvn clean install -q -pl ":$$modules" -P"$$profile" -am -T8 -Dmaven.test.skip=true -Dfederate.packaging=true) || exit 1; \
		done; \
	else \
		echo "☕️ Skipped Components Installation"; \
	fi
	FUSION_STARTER_DIR=$$(dirname $(1)); \
	echo "☕️ Local installing fusion-starter.jar on $$FUSION_STARTER_DIR ..."; \
	$(MAKE) -C $$FUSION_STARTER_DIR install; \
	echo "☕️ Optimizing target project"; \
	federate microservice optimize -i $(1) --yes; \
	echo "🍺 Congrat, consolidated!"
endef
