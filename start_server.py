import http.server
import socketserver
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(current_dir, 'static')
os.chdir(static_dir)

PORT = 8000
Handler = http.server.SimpleHTTPRequestHandler

with socketserver.TCPServer(('127.0.0.1', PORT), Handler) as httpd:
    print(f"Serving at http://127.0.0.1:{PORT}")
    httpd.serve_forever()