import http.server
import socketserver
import os

def startServer():
    os.chdir('./autocolortool')
    PORT = 8080
    HOST = "0.0.0.0"
    Handler = http.server.SimpleHTTPRequestHandler
    httpd = socketserver.TCPServer((HOST, PORT), Handler)
    print("Serving at port", PORT)
    httpd.serve_forever()

if __name__ == '__main__':
    startServer()