import os
import sys
import subprocess

REQUIREMENTS_FILE = 'requirements.txt'
NOTEBOOK_FILE = 'notebooks/01_neural_network_basics.ipynb'


def install_packages():
    print('Installing required packages...')
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', REQUIREMENTS_FILE])
    print('All packages installed.')


def download_example_data():
    print('No external data required for Chapter 1. Skipping data download.')


def launch_notebook():
    print(f'Launching Jupyter notebook: {NOTEBOOK_FILE}')
    subprocess.run([sys.executable, '-m', 'notebook', NOTEBOOK_FILE])


def main():
    print('--- Neural Networks Book Setup ---')
    install_packages()
    download_example_data()
    print('\nSetup complete!')
    print('You can now run code examples and open the interactive notebook.')
    launch = input('Would you like to launch the main notebook now? (y/n): ').strip().lower()
    if launch == 'y':
        launch_notebook()
    else:
        print(f'You can launch it later with: jupyter notebook {NOTEBOOK_FILE}')


if __name__ == '__main__':
    main() 