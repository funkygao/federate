FROM registry.cn-hangzhou.aliyuncs.com/eryajf/golang:1.22.2-alpine3.19 AS builder

WORKDIR /app

COPY . .

# 设置环境变量
ENV CGO_ENABLED=0
ENV GOOS=linux
ENV GOARCH=amd64

# 获取Git信息
ARG GIT_COMMIT
ARG GIT_BRANCH
ARG GIT_STATE
ARG BUILD_DATE

# 构建应用
RUN go build -o federate \
    -ldflags " \
    -X 'federate/cmd/version.GitCommit=${GIT_COMMIT}' \
    -X 'federate/cmd/version.GitBranch=${GIT_BRANCH}' \
    -X 'federate/cmd/version.GitState=${GIT_STATE}' \
    -X 'federate/cmd/version.BuildDate=${BUILD_DATE}'"

# 最终阶段，只保留 /federate 这1个文件
FROM scratch

# 从构建阶段复制二进制文件
COPY --from=builder /app/federate /federate

# 使用这个镜像的其他 Dockerfile 可以这样引用 federate：
# FROM your-base-image
# COPY --from=federate:latest /federate /usr/local/bin/federate
