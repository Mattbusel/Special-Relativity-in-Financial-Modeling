#----------------------------------------------------------------
# Generated CMake target import file for configuration "MinSizeRel".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "srfm::srfm_momentum" for configuration "MinSizeRel"
set_property(TARGET srfm::srfm_momentum APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(srfm::srfm_momentum PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_MINSIZEREL "CXX"
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/lib/srfm_momentum.lib"
  )

list(APPEND _cmake_import_check_targets srfm::srfm_momentum )
list(APPEND _cmake_import_check_files_for_srfm::srfm_momentum "${_IMPORT_PREFIX}/lib/srfm_momentum.lib" )

# Import target "srfm::srfm_beta_calculator" for configuration "MinSizeRel"
set_property(TARGET srfm::srfm_beta_calculator APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(srfm::srfm_beta_calculator PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_MINSIZEREL "CXX"
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/lib/srfm_beta_calculator.lib"
  )

list(APPEND _cmake_import_check_targets srfm::srfm_beta_calculator )
list(APPEND _cmake_import_check_files_for_srfm::srfm_beta_calculator "${_IMPORT_PREFIX}/lib/srfm_beta_calculator.lib" )

# Import target "srfm::srfm_manifold" for configuration "MinSizeRel"
set_property(TARGET srfm::srfm_manifold APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(srfm::srfm_manifold PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_MINSIZEREL "CXX"
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/lib/srfm_manifold.lib"
  )

list(APPEND _cmake_import_check_targets srfm::srfm_manifold )
list(APPEND _cmake_import_check_files_for_srfm::srfm_manifold "${_IMPORT_PREFIX}/lib/srfm_manifold.lib" )

# Import target "srfm::srfm_geodesic" for configuration "MinSizeRel"
set_property(TARGET srfm::srfm_geodesic APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(srfm::srfm_geodesic PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_MINSIZEREL "CXX"
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/lib/srfm_geodesic.lib"
  )

list(APPEND _cmake_import_check_targets srfm::srfm_geodesic )
list(APPEND _cmake_import_check_files_for_srfm::srfm_geodesic "${_IMPORT_PREFIX}/lib/srfm_geodesic.lib" )

# Import target "srfm::srfm_engine" for configuration "MinSizeRel"
set_property(TARGET srfm::srfm_engine APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(srfm::srfm_engine PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_MINSIZEREL "CXX"
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/lib/srfm_engine.lib"
  )

list(APPEND _cmake_import_check_targets srfm::srfm_engine )
list(APPEND _cmake_import_check_files_for_srfm::srfm_engine "${_IMPORT_PREFIX}/lib/srfm_engine.lib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
