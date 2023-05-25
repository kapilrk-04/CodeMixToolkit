from flask import Flask, render_template, request, flash, url_for, jsonify
import re
import os
import subprocess
import sys
import threading



def popen_io(cmd):
#     p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1,
                         universal_newlines=True)
    def consume(s):
        for _ in s:
            pass
    threading.Thread(target=consume, args=(p.stderr,)).start()
    return p


# Simplified, non-threadsafe version for force_align.py
# Use the version in realtime for development
class Aligner:

    def __init__(self, fwd_params, fwd_err, rev_params, rev_err, heuristic='grow-diag-final-and'):

        build_root = os.path.dirname(os.path.abspath(__file__))
        fast_align = os.path.join(build_root, 'fast_align')
        atools = os.path.join(build_root, 'atools')

        (fwd_T, fwd_m) = self.read_err(fwd_err)
        (rev_T, rev_m) = self.read_err(rev_err)

        fwd_cmd = [fast_align, '-i', '-', '-d', '-T', fwd_T, '-m', fwd_m, '-f', fwd_params]
        rev_cmd = [fast_align, '-i', '-', '-d', '-T', rev_T, '-m', rev_m, '-f', rev_params, '-r']
        tools_cmd = [atools, '-i', '-', '-j', '-', '-c', heuristic]

        self.fwd_align = popen_io(fwd_cmd)
        self.rev_align = popen_io(rev_cmd)
        self.tools = popen_io(tools_cmd)

    def align(self, line):
        print(type(line))
        print(line)
        self.fwd_align.stdin.write('{}\n'.format(line))
        self.rev_align.stdin.write('{}\n'.format(line))
        # f words ||| e words ||| links ||| score
        fwd_line = self.fwd_align.stdout.readline().split('|||')[2].strip()
        rev_line = self.rev_align.stdout.readline().split('|||')[2].strip()
        self.tools.stdin.write('{}\n'.format(fwd_line))
        self.tools.stdin.write('{}\n'.format(rev_line))
        al_line = self.tools.stdout.readline().strip()
        return al_line
 
    def close(self):
        self.fwd_align.stdin.close()
        self.fwd_align.wait()
        self.rev_align.stdin.close()
        self.rev_align.wait()
        self.tools.stdin.close()
        self.tools.wait()

    def read_err(self, err):
        (T, m) = ('', '')
        for line in open(err):
            # expected target length = source length * N
            if 'expected target length' in line:
                m = line.split()[-1]
            # final tension: N
            elif 'final tension' in line:
                T = line.split()[-1]
        return (T, m)

fwd_params = "fwd_params"
fwd_err = "fwd_err"
rev_params = "rev_params"
rev_err = "rev_err"
heuristic = "grow-diag-final-and"

args = [fwd_params, fwd_err, rev_params, rev_err, heuristic]

aligner = Aligner(*args)

app = Flask(__name__)
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


@app.route('/statistical_aligner_enhi', methods=['POST'])
def statistical_aligner_enhi():

    req_data = request.get_json()
    l1 = req_data['l1']
    l2 = req_data['l2']
    
    line = f"{l1} ||| {l2}"
    
    alignment = aligner.align(line.strip())


    d = {"l1" : l1,
        "l2" : l2,
        "alignment" : alignment}

    return jsonify(d)




if __name__ == '__main__':
    app.run(host='0.0.0.0', port = 6000, debug=True)#can be changed by user if this to be run on another port.
