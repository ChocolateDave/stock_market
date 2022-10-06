from distutils.core import setup
from src.version import __version__

setup(
    name="stock_market",
    version=__version__,
    author=["Maverick Zhang maverickzhang@berkeley.edu",
            "Juanwu Lu juanwu_lu@berkeley.edu"],
    packages=[
        "agents",
        "infrastructure",
        "nn",
        "policy",
        "utils"
    ],
    package_dir={
        "agents": "src/agents",
        "infrastructure": "src/infrastructure",
        "nn": "src/nn",
        "policy": "src/policy",
        "utils": "src/utils"
    }
)
