import os.path
import pybind11
import sys
import argh


def main(geant4_path=None):
    dir = pybind11.__file__
    dir = os.path.join(os.path.split(dir)[0], 'share/cmake/pybind11')

    print("Using python:", sys.executable)
    print("pybind11 found in:", dir)

    if geant4_path:
        print("Using geant4 path:", geant4_path)
        print("The following cmake command can be used:\n"
              "cmake -Dpybind11_DIR=%s -DPython_EXECUTABLE=%s"
              " -DCMAKE_PREFIX_PATH=%s -DGEANT4_INCLUDE_DIR=%s"
              " .."% (dir, sys.executable, geant4_path, os.path.join(geant4_path, 'include/geant4')))


    else:
        print("The following cmake command can be used:\n"
              "cmake -Dpybind11_DIR=%s -DPython_EXECUTABLE=%s .." % (dir, sys.executable))


if __name__ == "__main__":
    argh.dispatch_command(main)
