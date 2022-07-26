from flask import Flask, send_file, send_from_directory, request
import os
import subprocess

def read_file(address):
    res = ""
    with open(address) as f:
        res = f.read()
    return res

def process_image(filename, device, threshold1, threshold2, brightness_change):
    if not device:
        return read_file('ui/error.html').format("Please choose a valid device.")
    device = device.lower()
    if device not in ['cpu', 'cuda']:
        return read_file('ui/error.html').format("Please choose a valid device.")
    origin_path = os.path.join('.', 'Test_Images', filename)
    if not os.path.isfile(origin_path):
        return read_file('ui/error.html').format("Please specify a valid image in the 'Test_Images' folder.")
    brightness_result_path = os.path.join('.', 'Result_Images', f'{device}_output_brightness_{filename}')
    result_path = os.path.join('.', 'Result_Images', f'{device}_output_{filename}')
    if device == 'cuda':
        args = f'{origin_path} {brightness_result_path} {result_path} {threshold1} {threshold2} {brightness_change}'
    else:
        args = f'{origin_path} {result_path} {threshold1} {threshold2} {brightness_change}'
    COMMANDS = {
        'cpu': ['g++ -O2 -Wall -std=c++11 Code.cpp `pkg-config --cflags --libs opencv4` && ./a.out ' + args],
        'cuda': ['nvcc Code.cu `pkg-config --cflags --libs opencv4` -o cuda.out && ./cuda.out ' + args]
    }
    elapsed = subprocess.Popen(COMMANDS[device], stdout=subprocess.PIPE, shell=True).stdout.read().decode('utf-8')
    res = read_file(f'ui/{device}.html')
    if device == 'cuda':
        res = res.format(device.upper(), elapsed, origin_path, brightness_result_path, result_path)
    else:
        res = res.format(device.upper(), elapsed, origin_path, result_path)
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
    filename = os.path.basename(filename)
    return process_image(
        filename, 
        request.form.get('type'), 
        request.form.get('threshold_1'), 
        request.form.get('threshold_2'), 
        request.form.get('brightness_change')
    )

@app.route("/Test_Images/<path:path>")
def serve_test_image(path):
    return send_from_directory('Test_Images', path)

@app.route("/Result_Images/<path:path>")
def serve_result_image(path):
    return send_from_directory('Result_Images', path)

@app.route("/style.css")
def serve_style():
    return send_file("ui/style.css")