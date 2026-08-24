#!/bin/bash

set -e

# PEP 668: the sist2 Docker image is Ubuntu-based, where pip refuses to install into the
# system environment without --break-system-packages. Older pip does not know the flag.
if pip install --help | grep -q -- --break-system-packages; then
    pip install --break-system-packages -r requirements.txt
else
    pip install -r requirements.txt
fi
