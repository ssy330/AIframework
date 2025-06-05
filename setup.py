from setuptools import setup, find_packages

setup(
    name='dezero_mlp',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'dezero',  # 의존성 패키지
    ],
    author='SoYoon Park',
    description='MLP Model using DeZero Framework',
)