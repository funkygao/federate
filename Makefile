SHELL := /bin/bash
.SILENT:

GIT_COMMIT := $(shell git rev-parse --short HEAD)
GIT_BRANCH := $(shell git rev-parse --abbrev-ref HEAD)
GIT_STATE := $(shell if git diff-index --ignore-submodules=all --quiet HEAD --; then echo "clean"; else echo "dirty"; fi)
BUILD_DATE := $(shell LC_TIME=zh_CN.UTF-8 date +"%A %Y/%m/%d %H:%M:%S")

help:
	awk 'BEGIN {FS = ":.*##"; printf "Usage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-30s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } /^##[^@]/ { printf "%s\n", substr($$0, 4) }' $(MAKEFILE_LIST)

##@ Build

generate:
	go generate ./...

fmt:
	go fmt ./...

test: generate fmt
	go test ./...

clean:
	find . \( -name prompt.txt -o -name .DS_Store \) -exec rm -f {} \;
	rm -rf build

install: test ## Build and install federate. If HOMEBREW_PREFIX is set, install there, otherwise use GOPATH/bin.
	if [ -n "$(HOMEBREW_PREFIX)" ]; then \
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

install-completion: ## Install shell completion for federate on MacOS.
	if [ "$$(uname)" != "Darwin" ]; then \
		echo "This target is only for MacOS users."; \
		exit 1; \
	fi; \
	SHELL_NAME=$$(basename "$$SHELL"); \
	echo "Detected SHELL: $$SHELL_NAME"; \
	case $$SHELL_NAME in \
		bash) \
			if ! command -v brew >/dev/null 2>&1; then \
				echo "Homebrew is required to install bash-completion. Please install Homebrew first."; \
				exit 1; \
			fi; \
			if ! brew list bash-completion >/dev/null 2>&1; then \
				echo "Installing bash-completion..."; \
				brew install bash-completion; \
			fi; \
			COMPLETION_DIR=$$(brew --prefix)/etc/bash_completion.d; \
			federate completion bash > "$$COMPLETION_DIR/federate"; \
			echo "Bash completion for federate installed to $$COMPLETION_DIR/federate"; \
			echo "Please add the following line to your ~/.bash_profile or ~/.bashrc:"; \
			echo "  . $$(brew --prefix)/etc/profile.d/bash_completion.sh"; \
			;; \
		zsh) \
			COMPLETION_DIR=$$HOME/.zsh/completion; \
			mkdir -p "$$COMPLETION_DIR"; \
			federate completion zsh > "$$COMPLETION_DIR/_federate"; \
			echo "Zsh completion for federate installed to $$COMPLETION_DIR/_federate"; \
			echo "Please add the following lines to your ~/.zshrc:"; \
			echo "  fpath=($$COMPLETION_DIR \$$fpath)"; \
			echo "  autoload -Uz compinit && compinit"; \
			;; \
		fish) \
			COMPLETION_DIR=$$HOME/.config/fish/completions; \
			mkdir -p "$$COMPLETION_DIR"; \
			federate completion fish > "$$COMPLETION_DIR/federate.fish"; \
			echo "Fish completion for federate installed to $$COMPLETION_DIR/federate.fish"; \
			;; \
		*) \
			echo "Unsupported shell: $$SHELL_NAME"; \
			exit 1; \
			;; \
	esac; \
	echo "Completion installation finished. Please restart your shell or source the appropriate file to enable completion."

docker-build: ## Build Docker image for federate.
	docker build \
		--build-arg GIT_COMMIT=$(GIT_COMMIT) \
		--build-arg GIT_BRANCH=$(GIT_BRANCH) \
		--build-arg GIT_STATE=$(GIT_STATE) \
		--build-arg BUILD_DATE='$(BUILD_DATE)' \
		-t federate:latest .
	echo "🍺 Docker image built: federate:latest"

docker-test: docker-build
	echo "Creating temporary Dockerfile for testing..."
	echo "FROM centos:7" > Dockerfile.test
	echo "COPY --from=federate:latest /federate /usr/local/bin/federate" >> Dockerfile.test
	echo "Building test image..."
	docker build -t federate:test -f Dockerfile.test .
	echo "Running test container..."
	docker run --rm federate:test ls -l /usr/local/bin/federate
	echo "Cleaning up..."
	rm -f Dockerfile.test
	docker rmi federate:test

PLATFORMS := linux-amd64 # linux-arm64 darwin-amd64 darwin-arm64

release: ## Build binaries for darwin-amd64, darwin-arm64, linux-amd64 & linux-arm64.
	for platform in $(PLATFORMS); do \
		GOOS=$${platform%%-*} GOARCH=$${platform##*-} go build \
		-o build/federate-$$platform \
		-ldflags " \
			-X 'federate/cmd/version.GitCommit=$(GIT_COMMIT)' \
			-X 'federate/cmd/version.GitBranch=$(GIT_BRANCH)' \
			-X 'federate/cmd/version.GitState=$(GIT_STATE)' \
			-X 'federate/cmd/version.BuildDate=$(BUILD_DATE)'"; \
	done
