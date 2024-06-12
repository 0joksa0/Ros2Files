from setuptools import find_packages, setup

package_name = 'my_drone_controll'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'opencv-python-headless',
        'geopy',
    ],
    zip_safe=True,
    maintainer='aleksandar',
    maintainer_email='93257871+0joksa0@users.noreply.github.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
           'drone_controller = my_drone_controll.drone_controller:main',
           'humidity_sensor = my_drone_controll.sensor:main',
        ],
    },
)
