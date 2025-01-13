SHELL := /bin/bash
.DEFAULT_GOAL := help
.SILENT:

MAVEN = mvn

# Variables to be built into the binary
GIT_COMMIT  := $(shell git rev-parse --short HEAD)
GIT_BRANCH  := $(shell git rev-parse --abbrev-ref HEAD)
GIT_USER    := $(shell git config user.name)
GIT_STATE   := $(shell if git diff-index --ignore-submodules=all --quiet HEAD --; then echo "clean"; else echo "dirty"; fi)
BUILD_DATE  := $(shell LC_TIME=zh_CN.UTF-8 date +"%A %Y/%m/%d %H:%M:%S")

# go -ldflags
LDFLAGS := -X 'federate/cmd/version.GitUser=$(GIT_USER)' \
           -X 'federate/cmd/version.GitCommit=$(GIT_COMMIT)' \
           -X 'federate/cmd/version.GitBranch=$(GIT_BRANCH)' \
           -X 'federate/cmd/version.GitState=$(GIT_STATE)' \
           -X 'federate/cmd/version.BuildDate=$(BUILD_DATE)'

# go build packages
EXCLUDE_PACKAGES := /internal/plugin /internal/algo internal/javast/
PACKAGES := $(shell go list ./... | grep -Ev '$(subst $() ,|,$(EXCLUDE_PACKAGES))')

# go build tags
INCLUDE_PROGUARD ?= 0
ifeq ($(INCLUDE_PROGUARD),1)
    BUILD_TAGS += proguard
endif

help:
	awk 'BEGIN {FS = ":.*##"; printf "Usage:\n  make \033[36m<target> INCLUDE_PROGUARD=[0*|1]\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } /^##[^@]/ { printf "%s\n", substr($$0, 4) }' $(MAKEFILE_LIST)

##@ Build

vet:
	go vet ./...

fmt:
	go fmt ./...

race: fmt
	go test -race $(PACKAGES)

test: fmt
	go test $(PACKAGES)

stress:
	go test $(PACKAGES) -test.count 20 -test.failfast

coverage:
	go test $(PACKAGES) -cover | grep -w ok | column -t

clean:
	git clean -df
	cd $(JAVAST_DIR) && $(MAVEN) clean -q

install:embed-javast test ## Check if Go is installed, install if not, then build and install federate.
	if ! command -v go >/dev/null 2>&1; then \
		echo "Golang is not installed. Attempting to install via Homebrew..."; \
		if ! command -v brew >/dev/null 2>&1; then \
			echo "Error: Homebrew is not installed. Please install Homebrew first."; \
			exit 1; \
		fi; \
		brew install go; \
	fi
	if [ -n "$(HOMEBREW_PREFIX)" ]; then \
		go build -o $(HOMEBREW_PREFIX)/bin/federate -ldflags "$(LDFLAGS)" $(PACKAGES); \
		echo "🍺 Installed to $(HOMEBREW_PREFIX)/bin/federate"; \
	else \
		go install -tags "$(BUILD_TAGS)" -ldflags "$(LDFLAGS)" $(PACKAGES); \
		echo "🍺 Installed to $$(go env GOPATH)/bin/federate"; \
	fi

quick-install:fmt
	go install -tags "$(BUILD_TAGS)" -ldflags "$(LDFLAGS)" $(PACKAGES)
	echo "🍺 Installed to $$(go env GOPATH)/bin/federate"

completion: ## Install shell completion for federate on MacOS.
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
			echo "+---------------------------------------------------------+"; \
			echo "| Please add the following lines to your ~/.zshrc:        |"; \
			echo "+---------------------------------------------------------+"; \
			echo "fpath=($$COMPLETION_DIR \$$fpath)"; \
			echo "autoload -Uz compinit && compinit"; \
			echo "+---------------------------------------------------------+"; \
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
	echo "🍺 Completion installation finished. Please restart your shell or source the appropriate file to enable completion."

# Docker

docker-build:
	docker build \
		--build-arg GIT_USER=$(GIT_USER) \
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

PLATFORMS := linux-amd64 linux-arm64 darwin-amd64 darwin-arm64

release:embed-javast
	for platform in $(PLATFORMS); do \
		GOOS=$${platform%%-*} GOARCH=$${platform##*-} go build \
		-o build/federate-$$platform \
		-ldflags "$(LDFLAGS)"; \
	done

# Diagnose

PROFILE_DURATION=30
PPROF_PORT=9087
PROFILE_FILE=cpu_profile.pb.gz
FLAMEGRAPH_DIR=~/github/FlameGraph

loc:
	for dir in pkg/* cmd/*; do \
		if [ -d "$$dir" ]; then \
			loc=$$(cloc $$dir --include-lang=Go --json | jq -r '.Go.code // 0'); \
			if [ $$loc -gt 200 ]; then \
				echo "$$dir,$$loc" >> .code_stats.csv; \
			fi \
		fi \
	done
	sort -t',' -k2 -nr .code_stats.csv | column -t -s','
	rm -f .code_stats.csv

profile:
	go tool pprof -seconds=$(PROFILE_DURATION) -proto http://localhost:$(PPROF_PORT)/debug/pprof/profile > $(PROFILE_FILE)

pprof:profile
	go install github.com/google/pprof@latest
	pprof -http=:8080 $(PROFILE_FILE)

flamegraph:profile
	echo "git clone https://github.com/brendangregg/FlameGraph.git"
	go tool pprof -raw -output=cpu_profile.txt $(PROFILE_FILE)
	$(FLAMEGRAPH_DIR)/stackcollapse-go.pl cpu_profile.txt > cpu_profile.folded
	$(FLAMEGRAPH_DIR)/flamegraph.pl cpu_profile.folded > cpu_flamegraph.svg
	echo "🍺 Checkout cpu_flamegraph.svg"

# Java AST Transformer
JAVAST_DIR = internal/javast
JAVAST_JAR= $(JAVAST_DIR)/target/javast.jar
EMBED_DIR = internal/fs/templates/jar

embed-javast:
	echo "Java AST Transformer packaging ..."
	cd $(JAVAST_DIR) && $(MAVEN) clean package -q
	mkdir -p $(EMBED_DIR)/
	mv -f $(JAVAST_JAR) $(EMBED_DIR)/
	echo "Java AST Transformer embedded"
