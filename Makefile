.PHONY: all clean release install completion-bash fmt test

GIT_COMMIT := $(shell git rev-parse --short HEAD)
GIT_BRANCH := $(shell git rev-parse --abbrev-ref HEAD)
GIT_STATE := $(shell if git diff-index --ignore-submodules=all --quiet HEAD --; then echo "clean"; else echo "dirty"; fi)
BUILD_DATE := $(shell LC_TIME=zh_CN.UTF-8 date +"%A %Y/%m/%d %H:%M:%S")

all: install

fmt:
	@go fmt ./...

test: fmt
	@go test ./...

clean:
	@rm -f federate-darwin-*
	@find . \( -name prompt.txt -o -name .DS_Store \) -exec rm -f {} \;

install: test
	@go install -ldflags "\
		-X 'federate/cmd.GitCommit=$(GIT_COMMIT)' \
		-X 'federate/cmd.GitBranch=$(GIT_BRANCH)' \
		-X 'federate/cmd.GitState=$(GIT_STATE)' \
		-X 'federate/cmd.BuildDate=$(BUILD_DATE)'"

completion-bash: install
	@federate completion bash > /usr/local/etc/bash_completion.d/federate

release:
	@GOOS=darwin GOARCH=amd64 go build -o federate-darwin-amd64 -ldflags "\
		-X 'federate/cmd.GitCommit=$(GIT_COMMIT)' \
		-X 'federate/cmd.GitBranch=$(GIT_BRANCH)' \
		-X 'federate/cmd.GitState=$(GIT_STATE)' \
		-X 'federate/cmd.BuildDate=$(BUILD_DATE)'"
	@GOOS=darwin GOARCH=arm64 go build -o federate-darwin-arm64 -ldflags "\
		-X 'federate/cmd.GitCommit=$(GIT_COMMIT)' \
		-X 'federate/cmd.GitBranch=$(GIT_BRANCH)' \
		-X 'federate/cmd.GitState=$(GIT_STATE)' \
		-X 'federate/cmd.BuildDate=$(BUILD_DATE)'"
