from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

ext = [Extension("cy_qei", ["cy_qei.pyx"])]

setup(name="cython qei",
      cmdclass={'build_ext': build_ext},
      include_dirs=[np.get_include()],
      ext_modules=ext)
