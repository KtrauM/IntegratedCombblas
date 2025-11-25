# Top level directory has the include files


ADD_EXECUTABLE( tdbfs TopDownBFS.cpp )
ADD_EXECUTABLE( ccbfs ConnectedComponentBFS.cpp )
ADD_EXECUTABLE( dobfs DirOptBFS.cpp )
ADD_EXECUTABLE( fbfs FilteredBFS.cpp )
ADD_EXECUTABLE( fmis FilteredMIS.cpp )
ADD_EXECUTABLE( mcl MCL.cpp )
ADD_EXECUTABLE( betwcent BetwCent.cpp )
ADD_EXECUTABLE( lacc CC.cpp)
add_library(lacc-lib INTERFACE)
target_sources(lacc-lib INTERFACE
  FILE_SET HEADERS
  FILES
  CC.h
)
target_link_libraries(lacc-lib INTERFACE CombBLAS)
target_link_libraries(lacc-lib INTERFACE CombBLAS)
target_link_libraries(lacc PRIVATE lacc-lib)

ADD_EXECUTABLE( fastsv FastSV.cpp)
add_library(fastsv-lib INTERFACE)
target_sources(fastsv-lib INTERFACE
  FILE_SET HEADERS
  FILES
  FastSV.h
)
target_link_libraries(fastsv-lib INTERFACE CombBLAS)
TARGET_LINK_LIBRARIES(fastsv PRIVATE fastsv-lib)

TARGET_LINK_LIBRARIES( tdbfs CombBLAS KaGen::KaGen)
TARGET_LINK_LIBRARIES( ccbfs CombBLAS KaGen::KaGen)
TARGET_LINK_LIBRARIES( dobfs CombBLAS)
TARGET_LINK_LIBRARIES( fbfs CombBLAS)
TARGET_LINK_LIBRARIES( fmis CombBLAS)
TARGET_LINK_LIBRARIES( mcl CombBLAS)
TARGET_LINK_LIBRARIES( betwcent CombBLAS)


ADD_TEST(NAME BetwCent_Test COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:betwcent> ../TESTDATA/SCALE16BTW-TRANSBOOL/ 10 96 )
ADD_TEST(NAME TopDownBFS_Test COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:tdbfs> Force 17 FastGen)
ADD_TEST(NAME DirOptBFS_Test COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:dobfs> 17 )
ADD_TEST(NAME FBFS_Test COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:fbfs> Gen 16 )
ADD_TEST(NAME FMIS_Test COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:fmis> 17 )
