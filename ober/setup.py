from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

ext_modules = []
ext_modules.append(Extension("ober.tokens.tokens_inner", ["ober/tokens/tokens_inner.pyx"], include_dirs=[numpy.get_include()]))

setup(
    name="ober-ml",
    packages=["ober",
              "ober.documents", 
              "ober.tokens", 
              "ober.senses",
              "ober.models", 
              "ober.models.phrases",
              "ober.models.sense2vec",
              "ober.models.word2vec"],
    ext_modules=cythonize(ext_modules)
)