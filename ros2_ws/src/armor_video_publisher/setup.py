from setuptools import find_packages, setup

package_name = "armor_video_publisher"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    description="Publish armor test video frames as a ROS 2 image topic.",
    entry_points={
        "console_scripts": [
            "video_publisher = armor_video_publisher.video_publisher:main",
        ],
    },
)
