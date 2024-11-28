def setup():
    import subprocess

    subprocess.run(['chmod', '+x', '~/project/install.sh'])
    subprocess.run(['~/project/install.sh'], shell=True)

