from setuptools import setup
from setuptools import find_packages

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='behind_the_stars',
      version="0.1",
      description= "Predicting restaurants performance based on reviews",
      author="Best team of Le Wagon #2195 batch",
      install_requires=requirements,
      packages=find_packages(),
      include_package_data=True)
