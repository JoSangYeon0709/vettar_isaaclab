"""pyproject.toml 만으로 설치되지만, 일부 환경(omni.kit extension loader)이
setup.py 를 기대하므로 호환용 stub 만 둔다."""

from setuptools import setup

setup()
