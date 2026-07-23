# Small image with a POSIX shell: the agent only needs `bash`/`sh` reachable
# through environment.exec. Alpine keeps the CI pull fast.
FROM alpine:3.20

RUN apk add --no-cache bash coreutils
WORKDIR /app
