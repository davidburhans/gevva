#!/usr/bin/env python3
"""Lightweight HTTP server to view the Gevva technical deep-dive website locally.
Usage:
    uv run python scripts/serve_website.py [--port 8080]
"""

import argparse
import http.server
import socketserver
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Serve the Gevva technical explainer website.")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on (default: 8080)")
    parser.add_argument("--dir", type=str, default="website", help="Directory to serve (default: website)")
    args = parser.parse_args()

    serve_dir = Path(args.dir).resolve()
    if not serve_dir.exists():
        serve_dir = Path("spaces/gevva-static").resolve()

    os.chdir(serve_dir)

    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", args.port), handler) as httpd:
        print(f"===========================================================")
        print(f"⚡ Gevva Technical Website Server Running")
        print(f"🌐 Local URL: http://localhost:{args.port}/")
        print(f"📁 Serving Directory: {serve_dir}")
        print(f"💡 Press Ctrl+C to stop the server.")
        print(f"===========================================================")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")

if __name__ == "__main__":
    main()
