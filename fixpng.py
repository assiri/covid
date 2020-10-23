import os
import subprocess

def system_call(args, cwd="."):
    print("Running '{}' in '{}'".format(str(args), cwd))
    subprocess.call(args, cwd=cwd,shell=True)
    pass

def fix_image_files(root=os.curdir):
    for path, dirs, files in os.walk(os.path.abspath(root)):
    
        # sys.stdout.write('.')
        for dir in dirs:
            print( "{}".format(os.path.join(path, dir)))
            system_call("/usr/local/bin/mogrify **/*.png", "{}".format(os.path.join(path, dir)))


fix_image_files(os.curdir)
