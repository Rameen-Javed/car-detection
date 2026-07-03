import http.server
import socketserver
import webbrowser
import threading
import time
import sys

PORT = 8000
Handler = http.server.SimpleHTTPRequestHandler

def open_browser():
    # Wait a brief moment to ensure the server starts listening
    time.sleep(1.2)
    url = f"http://localhost:{PORT}"
    print(f"Opening browser to {url}...")
    webbrowser.open(url)

def run_server():
    global PORT
    # Allow port reuse to avoid 'address already in use' errors if restarted quickly
    socketserver.TCPServer.allow_reuse_address = True
    
    while True:
        try:
            with socketserver.TCPServer(("", PORT), Handler) as httpd:
                print(f"\n=======================================================")
                print(f"  AutoDetect AI Server successfully running on Port {PORT}")
                print(f"  Url: http://localhost:{PORT}")
                print(f"  To stop the server, press Ctrl+C in this console.")
                print(f"=======================================================\n")
                
                # Start browser opening thread
                threading.Thread(target=open_browser, daemon=True).start()
                
                httpd.serve_forever()
        except OSError as e:
            # If port 8000 is in use, try a different port automatically
            if e.errno == 98 or e.errno == 10048: # Port already in use error codes
                print(f"Port {PORT} is already in use. Trying port {PORT + 1}...")
                PORT += 1
            else:
                raise e
        except KeyboardInterrupt:
            print("\nShutting down server.... Goodbye!")
            sys.exit(0)

if __name__ == "__main__":
    run_server()
