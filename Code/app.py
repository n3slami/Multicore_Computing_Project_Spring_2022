from flask import Flask, send_file, send_from_directory, request
import os
import subprocess

def read_file(address):
    res = ""
    with open(address) as f:
        res = f.read()
    return res

app = Flask('Multicore')

@app.route("/")
def serve_main():
    return send_file("ui/index.html")

@app.route("/result", methods=['GET', 'POST'])
def serve_result():
    filename = request.form.get('myfile')
    if not filename:
        return read_file('ui/error.html').format("Please specify the target image.")
    origin_path = os.path.join('.', 'Test_Images', filename)
    if not os.path.isfile(origin_path):
        return read_file('ui/error.html').format("Please specify a valid image in the 'Test_Images' folder.")
    result_path = os.path.join('.', 'Result_Images', 'output_' + os.path.basename(filename))

    if request.form.get('type') == 'CPU':
        command = ['g++ -O2 -Wall -std=c++11 Code.cpp `pkg-config --cflags --libs opencv4` && ./a.out ' + origin_path + ' ' + result_path]
        elapsed = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True).stdout.read().decode('utf-8')
        res = read_file('ui/cpu.html')
        res = res.format(elapsed, origin_path, result_path)
        return res
    elif request.form.get('type') == 'GPU':
        command = ['nvcc Code.cu `pkg-config --cflags --libs opencv4` -o cuda.out && ./cuda.out ' + origin_path]
        elapsed = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True).stdout.read().decode('utf-8')
        res = read_file('ui/gpu.html')
        res = res.format(elapsed, origin_path)
        return res
    else:
        return read_file('ui/error.html').format("Please try again.")

@app.route("/Test_Images/<path:path>")
def serve_test_image(path):
    return send_from_directory('Test_Images', path)

@app.route("/Result_Images/<path:path>")
def serve_result_image(path):
    return send_from_directory('Result_Images', path)

@app.route("/style.css")
def serve_style():
    return send_file("ui/style.css")