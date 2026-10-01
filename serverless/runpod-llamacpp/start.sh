#!/bin/bash
# Entrypoint for serverless/runpod-llamacpp/Dockerfile: runs llama-server on a
# loopback port and nginx on $PORT in front of it, mapping Runpod's `/ping` health
# poll onto llama-server's `/health`. Exits as soon as either process does, so a
# crashed engine fails the worker visibly instead of leaving a dead endpoint up.
set -euo pipefail

LLAMA_PORT=8081

cat >/etc/nginx/nginx.conf <<EOF
worker_processes 1;
pid /tmp/nginx.pid;
error_log /dev/stderr warn;
events { worker_connections 64; }
http {
    access_log off;
    server {
        listen ${PORT};
        # Runpod's load balancer: 200 = ready, 204 = still initializing.
        # llama-server answers /health with 503 while loading, and nginx itself
        # returns 502 before llama-server is listening at all.
        location = /ping {
            proxy_pass http://127.0.0.1:${LLAMA_PORT}/health;
            proxy_intercept_errors on;
            error_page 502 503 504 = @loading;
        }
        location = /healthz {
            proxy_pass http://127.0.0.1:${LLAMA_PORT}/health;
            proxy_intercept_errors on;
            error_page 502 503 504 = @loading;
        }
        location @loading { return 204; }
        location / {
            proxy_pass http://127.0.0.1:${LLAMA_PORT};
            proxy_http_version 1.1;
            proxy_buffering off;
            proxy_read_timeout 600s;
        }
    }
}
EOF

echo "starting llama-server: $GGUF_PATH ${LLAMA_ARGS}" >&2
# shellcheck disable=SC2086  # LLAMA_ARGS is a flag list and must word-split
/app/llama-server -m "$GGUF_PATH" --host 127.0.0.1 --port "$LLAMA_PORT" ${LLAMA_ARGS} &
nginx -g 'daemon off;' &

wait -n
echo "llama-server or nginx exited; stopping the worker" >&2
exit 1
