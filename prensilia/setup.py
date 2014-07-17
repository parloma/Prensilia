from os import getcwd, path, sep
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

CURRDIR = getcwd()

OPENCL_INC = "/opt/AMDAPP/include/"
OPENCL_LIBS = "/opt/AMDAPP/lib/x86/"

OPENNI_INC = "/usr/include/ni"


extensions = [
    Extension("pose_recognizer", ["pose_recognizer.pyx"],
              language="c++",
              include_dirs=[CURRDIR+sep+".."+sep+"pose_recognizer/",
                            CURRDIR+sep+".."+sep+"random_forest_gpu/",
                            OPENCL_INC],
              library_dirs=[CURRDIR+sep+".."+sep+"pose_recognizer"+sep+"build",
                            CURRDIR+sep+".."+sep+"random_forest_gpu"+sep+"build",
                            OPENCL_LIBS],
              libraries=["PoseRecognizer", "RandomForestGPU"]),
    Extension("hand_grabber", ["hand_grabber.pyx"],
              language="c++",
              include_dirs=[CURRDIR+sep+".."+sep+"hand_segmenter",
                            OPENNI_INC],
              library_dirs=[CURRDIR+sep+".."+sep+"hand_segmenter"+sep+"build"],
              libraries=["OpenNIHandSegmenter"])
    ]

setup(name = "IEIIT hand-pose recognition",
      ext_modules=cythonize(extensions))
