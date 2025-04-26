#!/bin/bash

git clone https://github.com/cc65/cc65.git
cd cc65
make

echo "export PATH=\"$(pwd)/cc65/bin:\$PATH\"" >> ~/.bashrc
echo "export CC65_HOME=\"$(pwd)/cc65\"" >> ~/.bashrc
source ~/.bashrc