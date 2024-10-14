define merge_project
	echo "☕️ Clearing up all generated target projects ..."
	$(MAKE) clean
	echo "☕️ Scaffolding target project fusion-starter ..."
	federate microservice fusion-start -i $(1)
	echo "☕️ Consolidating the target project ..."
	federate microservice consolidate -i $(1) --yes --yaml-conflict-cell-width 32 --silent=$(4)
	echo "☕️ Local installing the Rewritten Components ..."
	for repo in $$(federate components -i $(1)); do \
		profile=$$($(INVENTORY_CMD) -f maven-profile -r $$repo -e $(3)); \
		modules=$$($(INVENTORY_CMD) -f maven-modules -r $$repo); \
		echo "Installing $$repo with profile:$$profile on modules:$$modules"; \
		(cd $$repo && mvn clean install -q -pl ":$$modules" -P"$$profile" -am -T8 -Dmaven.test.skip=true -Dfederated.packaging=true) || exit 1; \
	done
	echo "☕️ Local installing fusion-starter.jar ..."
	$(MAKE) -C $(2) install
endef
