
from setuptools import setup, find_packages

with open("requirements.txt") as f:

    requirements = f.read().splitlines()

setup(

    name="flipkart-recommender",

    version="0.1.0",

    author="Maitreyi",

    author_email="kshmaitreyi18@gmail.com",

    description="A Flipkart product recommender system",

    packages=find_packages(),

    install_requires=requirements,

    python_requires=">=3.8",

)

