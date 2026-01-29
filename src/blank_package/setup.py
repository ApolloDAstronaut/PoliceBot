from setuptools import find_packages, setup

package_name = 'blank_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('lib/'+package_name,['blank_package/wheels.py']), #we added this line, add another if more nodes
        ('lib/'+package_name,['blank_package/camera.py']),
        ('lib/'+package_name,['blank_package/camera.py'])
 ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'wheels = blank_package.wheels:main',
            'blinker = blank_package.blinker:main',
            'camera = blank_package.camera:main',
        ],
    },

)
