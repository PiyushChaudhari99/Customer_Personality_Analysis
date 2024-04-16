from setuptools import find_packages,setup
from typing import List


hypen_e_dot = "-e ."
def get_requirments(file_path:str)->List[str]:
    """
        This function gets all the libraries inside the requirements file
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","")for req in requirements]

        if hypen_e_dot in requirements:
            requirements.remove(hypen_e_dot)

    return requirements



setup(
    name = "Customer_Personality_Analysis",
    version="0.0.1",
    author= "Piyush Chaudhari",
    author_email= "piyushpchaudhari@gamil.com",
    packages= find_packages(),
    install_requires = get_requirments('requirements.txt')

)