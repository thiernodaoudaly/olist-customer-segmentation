from setuptools import find_packages, setup

setup(
    name="olist-customer-segmentation",
    version="0.1.0",
    author="Thierno Daouda LY",              
    description="Segmentation des clients de Olist",       
    packages=find_packages(),   # trouve automatiquement tous les packages (dossiers avec __init__.py). On n'a pas à lister src, src.data, src.features... à la main.
    python_requires=">=3.9",
    install_requires=[],    
)