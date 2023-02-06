from setuptools import setup, find_packages

setup(
    name='morphological-manifold-mapping',
    version='0.0.1',
    description='Models and visually displays the geometry of a Fischer information matrix',
    author='Tensornetics LLC',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.0.0',
        'matplotlib>=3.0.0',
        'scipy>=1.0.0',
        'torch>=1.0.0',
        'tensorflow>=2.0.0',
    ]
)