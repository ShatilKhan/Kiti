"""
Setup script for Optical Flow Project
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / 'README.md'
long_description = ''
if readme_file.exists():
    long_description = readme_file.read_text(encoding='utf-8')

# Read requirements
requirements_file = Path(__file__).parent / 'requirements.txt'
requirements = []
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='optical-flow-project',
    version='1.0.0',
    author='Kiti Team',
    author_email='',
    description='Optical flow video processing pipeline for autonomous vehicles',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ShatilKhan/Kiti',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.8',
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            # Note: Use run.py directly for development
            # 'optical-flow=src.main:main',
        ],
    },
    include_package_data=True,
    package_data={
        'src': ['config/*.py'],
    },
    keywords='optical-flow computer-vision yolo autonomous-vehicles object-detection',
)
