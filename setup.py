import os
import subprocess
import sys
from setuptools import setup, find_packages
from setuptools.command.install import install as _install

VENV_DIR = 'venv'

def create_and_install():
    """Creates a virtual environment and installs dependencies."""
    if not os.path.exists(VENV_DIR):
        print(f"Creating virtual environment in {VENV_DIR}...")
        try:
            subprocess.check_call([sys.executable, '-m', 'venv', VENV_DIR])
            print("Virtual environment created successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error creating virtual environment: {e}")
            return False

    pip_executable = os.path.join(VENV_DIR, 'bin', 'pip')
    if sys.platform == 'win32':
        pip_executable = os.path.join(VENV_DIR, 'Scripts', 'pip.exe')

    try:
        subprocess.check_call([pip_executable, 'install', '-U', 'pip', 'setuptools'])
        subprocess.check_call([pip_executable, 'install', '-r', 'requirements.txt'])
        print("Dependencies installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False

# Custom install command to avoid showing usage message
class InstallCommand(_install):
    def run(self):
        _install.run(self)
        print("Package installed successfully.")

if __name__ == '__main__':
    if create_and_install():
        # Use a custom setup() call that includes the cmdclass parameter
        setup(
            name='cell-survival-rl',
            version='0.1.0',
            author='Barış Can Ataklı',
            author_email='bariscanatakli@posta.mu.edu.tr',
            description='A reinforcement learning-based simulation for virtual cell survival strategies.',
            packages=find_packages(where='src'),
            package_dir={'': 'src'},
            install_requires=[],  # Dependencies are handled by venv creation
            classifiers=[
                'Programming Language :: Python :: 3',
                'License :: OSI Approved :: MIT License',
                'Operating System :: OS Independent',
            ],
            python_requires='>=3.8',
            # This ensures we use our custom install command
            cmdclass={
                'install': InstallCommand,
            },
            script_args=['install', '--user'],  # Force the install command
        )
    else:
        print("Setup failed.")
        sys.exit(1)