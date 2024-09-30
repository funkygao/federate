.PHONY: help clean release install completion-bash

GIT_COMMIT := $(shell git rev-parse --short HEAD)
GIT_BRANCH := $(shell git rev-parse --abbrev-ref HEAD)
GIT_STATE := $(shell if git diff-index --ignore-submodules=all --quiet HEAD --; then echo "clean"; else echo "dirty"; fi)
BUILD_DATE := $(shell LC_TIME=zh_CN.UTF-8 date +"%A %Y/%m/%d %H:%M:%S")

help:
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-21s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Build

fmt:
	@go fmt ./...

test: fmt
	@go test ./...

clean: ## Clean up.
	@rm -f federate-darwin-*
	@find . \( -name prompt.txt -o -name .DS_Store \) -exec rm -f {} \;

install: test ## Build and install. If HOMEBREW_PREFIX is set, install there, otherwise use GOPATH/bin.
	@if [ -n "$(HOMEBREW_PREFIX)" ]; then \
		go build -o $(HOMEBREW_PREFIX)/bin/federate -ldflags "\
			-X 'federate/cmd/version.GitCommit=$(GIT_COMMIT)' \
			-X 'federate/cmd/version.GitBranch=$(GIT_BRANCH)' \
			-X 'federate/cmd/version.GitState=$(GIT_STATE)' \
			-X 'federate/cmd/version.BuildDate=$(BUILD_DATE)'"; \
		echo "Installed to $(HOMEBREW_PREFIX)/bin/federate"; \
	else \
		go install -ldflags "\
			-X 'federate/cmd/version.GitCommit=$(GIT_COMMIT)' \
			-X 'federate/cmd/version.GitBranch=$(GIT_BRANCH)' \
			-X 'federate/cmd/version.GitState=$(GIT_STATE)' \
			-X 'federate/cmd/version.BuildDate=$(BUILD_DATE)'"; \
		echo "Installed to $$(go env GOPATH)/bin/federate"; \
	fi

completion-bash: install ## Install bash completion for federate.
	@federate completion bash > /usr/local/etc/bash_completion.d/federate

PLATFORMS := darwin-amd64 darwin-arm64

release: ## Build binaries for darwin-amd64 & darwin-arm64.
	@for platform in $(PLATFORMS); do \
		GOOS=$${platform%%-*} GOARCH=$${platform##*-} go build \
		-o federate-$$platform \
		-ldflags " \
			-X 'federate/cmd/version.GitCommit=$(GIT_COMMIT)' \
			-X 'federate/cmd/version.GitBranch=$(GIT_BRANCH)' \
			-X 'federate/cmd/version.GitState=$(GIT_STATE)' \
			-X 'federate/cmd/version.BuildDate=$(BUILD_DATE)'"; \
	done
