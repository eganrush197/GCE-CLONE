# ABOUTME: Package setup configuration
# ABOUTME: Enables pip installation and CLI tool registration

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / 'README.md'
long_description = readme_file.read_text() if readme_file.exists() else ''

setup(
    name='gaussian-mesh-converter',
    version='0.1.0',
    description='Fast mesh-to-gaussian-splat converter using direct geometric conversion',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/gaussian-mesh-converter',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.24.0',
        'scipy>=1.11.0',
        'trimesh>=3.23.0',
        'pillow>=10.0.0',
    ],
    extras_require={
        'gpu': [
            'torch>=2.0.0',
            'torchvision>=0.15.0',
        ],
        'dev': [
            'pytest>=7.4.0',
            'pytest-cov>=4.1.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'mesh2gaussian=mesh_to_gaussian:cli_main',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Multimedia :: Graphics :: 3D Modeling',
        'Topic :: Scientific/Engineering :: Visualization',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    keywords='gaussian-splatting 3d-graphics mesh-conversion computer-graphics',
)

