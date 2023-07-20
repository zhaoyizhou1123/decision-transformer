from setuptools import setup, find_packages

setup(
        name='maze',
        version="0.0.1",
        description=(
            'maze environment'
        ),
        platforms=["all"],
        packages=find_packages(include=('maze','maze.*'))
    )
