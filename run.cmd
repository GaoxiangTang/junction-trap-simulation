@echo off
start /b jupyter.exe lab --certfile="" --keyfile="" --port=8888
start http://localhost:8888