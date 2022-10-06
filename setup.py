from distutils.core import setup
from src.version import __version__

setup(
    name="stock_market",
    version=__version__,
    author=["Maverick Zhang maverickzhang@berkeley.edu",
            "Juanwu Lu juanwu_lu@berkeley.edu"],
    packages=[
        "agent",
        "critic",
        "env",
        "memory",
        "nn",
        "policy",
        "trainer"
    ],
    package_dir={
        "agent": "src/agent",
        "critic": "src/critic",
        "env": "src/env",
        "memory": "src/memory",
        "nn": "src/nn",
        "policy": "src/policy",
        "trainer": "src/trainer"
    }
)
