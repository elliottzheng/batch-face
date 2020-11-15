from os import path as os_path

from setuptools import setup

import batch_face

this_directory = os_path.abspath(os_path.dirname(__file__))


# 读取文件内容
def read_file(filename):
    with open(os_path.join(this_directory, filename), encoding="utf-8") as f:
        long_description = f.read()
    return long_description


# 获取依赖
def read_requirements(filename):
    return [
        line.strip()
        for line in read_file(filename).splitlines()
        if not line.startswith("#")
    ]


setup(
    name="batch-face",
    version=batch_face.__version__,
    description="Batch Face Preprocessing for Modern Research",
    author="Elliott Zheng",
    author_email="admin@hypercube.top",
    url="https://github.com/elliottzheng/batch-face",
    license="MIT",
    keywords="face-detection pytorch RetinaFace face-alignment",
    project_urls={
        "Documentation": "https://github.com/elliottzheng/batch-face",
        "Source": "https://github.com/elliottzheng/batch-face",
        "Tracker": "https://github.com/elliottzheng/batch-face/issues",
    },
    long_description=read_file("README.md"),  # 读取的Readme文档内容
    long_description_content_type="text/markdown",  # 指定包文档格式为markdown
    packages=["batch_face"],
    install_requires=["numpy", "torch", "torchvision"],
)
