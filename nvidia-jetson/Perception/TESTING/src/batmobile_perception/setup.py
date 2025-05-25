from setuptools import find_packages, setup

import os
from glob import glob

package_name = 'batmobile_perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='aryan',
    maintainer_email='aryan100306@gmail.com',
    description='Joker detection node',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'publisher= batmobile_perception.pubv2:main',
            'subscriber= batmobile_perception.reciever_node:main',
            'locate_func = batmobile_perception.locate_func:main'  # Ensure locate_func is properly registered
        ],
    },
)
