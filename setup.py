from setuptools import setup, find_packages

setup(
    name="stress",
    version="0.1.1",
    packages=find_packages(),
    python_requires=">=3.9",
    setup_requires=["setuptools_scm"],
    install_requires=["tensorflow>=2.5.0", "toml>=0.10.2", "typer>=0.3.2"],
    extras_require={
        "dev": [
            "pip-tools",
            "pytest>=6.2.3",
            "black>=20.8b1",
        ]
    },
)
