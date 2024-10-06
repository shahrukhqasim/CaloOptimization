cmake_minimum_required(VERSION 3.12)
project(sampling_calo)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


set(PYBIND11_FINDPYTHON ON)

# Find pybind11
find_package(pybind11 REQUIRED)
find_package(Geant4 REQUIRED ui_all vis_all)
set(GEANT4_INCLUDE_DIR "/some/random/path" CACHE PATH "Path to Geant4 include directory")

include_directories(/usr/local/include/Geant4/)
include_directories(${GEANT4_INCLUDE_DIR})



# Add subdirectory for jsoncpp
add_subdirectory(jsoncpp)

# Add the include directory for jsoncpp
include_directories(jsoncpp/include)

set(SOURCE_FILES
        DetectorConstruction.cc
        PrimaryGeneratorAction.cc
        CustomSteppingAction.cc
        CustomEventAction.cc
        BoxyDetectorConstruction.cc
        SlimFilmSensitiveDetector.cc
        )


add_library(common_sources STATIC ${SOURCE_FILES})

add_executable(CaloVis main.cc)

# Create the Python module
target_link_libraries(CaloVis common_sources ${Geant4_LIBRARIES} jsoncpp_lib)
#target_link_libraries(sampling_calo ${Geant4_LIBRARIES})
set_target_properties(common_sources PROPERTIES POSITION_INDEPENDENT_CODE ON)


pybind11_add_module(sampling_calo HeterogeneousSamplingCalorimeter.cc)
target_link_libraries(sampling_calo PUBLIC common_sources ${Geant4_LIBRARIES} jsoncpp_lib)

configure_file(init_vis.mac init_vis.mac COPYONLY)
configure_file(design_alpha.json design_alpha.json COPYONLY)
