#!/usr/bin/env python3

import socket
import hashlib


def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()

    return ip


def get_md5(data):
    md5 = hashlib.md5()
    md5.update(data)
    return md5.hexdigest()