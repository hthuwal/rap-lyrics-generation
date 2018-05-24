from __future__ import print_function
import locale
import os
import argparse
import subprocess


def run_command(cmd, arguments):
    params = []
    params.append(cmd)
    if isinstance(arguments, list):
        params.extend(arguments)
    else:
        params.append(arguments)

    output = subprocess.check_output(params)

    # convert to string
    encoding = locale.getdefaultlocale()[1]
    output_str = output.decode(encoding)

    return output_str


def get_lyrics_stat(filename):
    cmd = 'bash'
    #jar_path = ".:analyze/copylibstask.jar:./weka.jar"
    cur_dir = os.getcwd()
    filename = os.path.abspath(filename)
    os.chdir("../analyze")
    arguments = ['run.sh', filename]
    output = run_command(cmd, arguments)
    os.chdir(cur_dir)
    result = {}
    for line in output.split('\n'):
        dv = line.split(':')

        if len(dv) == 2:
            key = dv[0].strip()
            value = float(dv[1].strip())
            result[key] = value

    return result


if __name__ == "__main__":
    # parse commandline arguments
    parser = argparse.ArgumentParser(description='Print statistics about lyrics')
    parser.add_argument("filename", help='Path to the file with lyrics')
    args = parser.parse_args()

    statistics = get_lyrics_stat(args.filename)
    for k in statistics.keys():
        print(k, statistics[k])
