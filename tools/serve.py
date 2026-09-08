#!/usr/bin/env python3
"""Build and serve only the generated static course on localhost."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import argparse
from build import main, OUT
class CourseHandler(SimpleHTTPRequestHandler):
    # .rs is registered as an unrelated XML format on some systems.
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map,
                      '.rs': 'text/plain; charset=utf-8',
                      '.toml': 'text/plain; charset=utf-8',
                      '.wgsl': 'text/plain; charset=utf-8',
                      '.md': 'text/plain; charset=utf-8'}
if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--port',type=int,default=8000); args=parser.parse_args()
    main()
    server=ThreadingHTTPServer(('127.0.0.1',args.port),partial(CourseHandler,directory=str(OUT)))
    print(f'Open http://127.0.0.1:{args.port}',flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: server.server_close()
