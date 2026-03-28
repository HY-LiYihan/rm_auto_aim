from setuptools import find_packages, setup

package_name = 'auto_aim_solver'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='venom',
    maintainer_email='liyihan.xyz@gmail.com',
    description='Ballistic prediction and command generation for rm_auto_aim.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ballistic_solver = auto_aim_solver.solver_node:main',
        ],
    },
)
