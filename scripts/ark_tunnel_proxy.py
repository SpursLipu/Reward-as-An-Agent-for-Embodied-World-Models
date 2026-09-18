"""Loopback CONNECT bridge to an SSH reverse forward; TLS stays end to end.

Requires ssh -R 127.0.0.1:17443:ark.cn-beijing.volces.com:443 on the client.
Only the Ark HTTPS destination is accepted. No credentials or payloads logged.
"""
import select
import socket
import socketserver
import time


class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        client = self.request
        client.settimeout(15)
        header = b''
        while b'\r\n\r\n' not in header:
            chunk = client.recv(4096)
            if not chunk:
                return
            header += chunk
            if len(header) > 16384:
                return
        lines, remaining = header.split(b'\r\n\r\n', 1)
        if lines.split(b'\r\n', 1)[0].split()[:2] != [b'CONNECT', b'ark.cn-beijing.volces.com:443']:
            client.sendall(b'HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n')
            return
        try:
            upstream = socket.create_connection(('127.0.0.1', 17443), timeout=15)
        except OSError:
            client.sendall(b'HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n')
            return
        with upstream:
            client.sendall(b'HTTP/1.1 200 Connection Established\r\n\r\n')
            client.settimeout(600)
            upstream.settimeout(600)
            if remaining:
                upstream.sendall(remaining)
            last = time.monotonic()
            try:
                while time.monotonic() - last < 600:
                    ready, _, _ = select.select([client, upstream], [], [], 10)
                    for src in ready:
                        data = src.recv(65536)
                        if not data:
                            return
                        (upstream if src is client else client).sendall(data)
                        last = time.monotonic()
            except (OSError, TimeoutError):
                return


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == '__main__':
    with Server(('127.0.0.1', 18081), Handler) as server:
        print('Ark-only CONNECT bridge listening on 127.0.0.1:18081', flush=True)
        server.serve_forever()
