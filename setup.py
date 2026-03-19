from setuptools import find_packages,setup
from typing import List 

HYPEN_E_DOT='-e .'
def get_requirements(file_paths:str)->List[str]:
    requirements=[]
    with open(file_paths) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]
        
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements

from setuptools import setup, find_packages

setup(
    name="mlproject",
    version="0.0.1",
    author="Nishit",
    author_email='nishitkhandhar26@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
    
)