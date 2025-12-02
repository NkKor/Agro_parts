# gunicorn.conf.py

import multiprocessing

bind = "0.0.0.0:3887"
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "sync"
worker_connections = 1000
timeout = 120
keepalive = 5

accesslog = "./logs/gunicorn_access.log"
errorlog = "./logs/gunicorn_error.log"
loglevel = "info"

preload_app = True
max_requests = 1000
max_requests_jitter = 100

limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190
