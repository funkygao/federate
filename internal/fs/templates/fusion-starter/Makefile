# Generated by federate, DO NOT EDIT.
SHELL := /bin/bash
.SILENT:

help:
	awk 'BEGIN {FS = ":.*##"; printf "Usage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } /^##[^@]/ { printf "%s\n", substr($$0, 4) }' $(MAKEFILE_LIST)

##@ Build

install: ## Compile and install {{.AppName}}-starter.jar on your host.
	mvn clean install -q -U -Dmaven.test.skip=true -T8
