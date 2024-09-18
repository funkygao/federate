# 使用多阶段构建来减少最终镜像的大小
FROM registry.cn-hangzhou.aliyuncs.com/eryajf/golang:1.22.2-alpine3.19 AS builder

WORKDIR /app

# 复制 go.mod 和 go.sum 并下载依赖
COPY go.mod go.sum ./
RUN go mod download

# 复制源代码
COPY . .

# 根据当前的芯片架构编译二进制文件
ARG TARGETARCH
RUN if [ "$TARGETARCH" = "x86_64" ]; then \
      CGO_ENABLED=0 GOOS=darwin GOARCH=amd64 go build -o federate; \
    else \
      CGO_ENABLED=0 GOOS=darwin GOARCH=arm64 go build -o federate; \
    fi

# 创建一个最小的运行时镜像
FROM alpine:latest

# 设置工作目录
WORKDIR /root/

# 复制编译好的二进制文件
COPY --from=builder /app/federate /usr/local/bin/federate

# 设置默认命令
ENTRYPOINT ["cp", "/usr/local/bin/federate", "/mac/federate"]
