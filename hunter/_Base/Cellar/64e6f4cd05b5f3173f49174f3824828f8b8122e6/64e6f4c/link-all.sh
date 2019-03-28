export "HUNTER_CELLAR_RAW_DIRECTORY=/home/fshan/Projects/LevelSetFusion2D/LevelSetFusion2DExperiments/cpp/hunter/_Base/Cellar/64e6f4cd05b5f3173f49174f3824828f8b8122e6/64e6f4c/raw"

ln \
  "${HUNTER_CELLAR_RAW_DIRECTORY}/lib/cmake/Boost/BoostConfig.cmake" \
  "$1/lib/cmake/Boost/BoostConfig.cmake"

ln \
  "${HUNTER_CELLAR_RAW_DIRECTORY}/lib/cmake/Boost/BoostConfigVersion.cmake" \
  "$1/lib/cmake/Boost/BoostConfigVersion.cmake"

ln \
  "${HUNTER_CELLAR_RAW_DIRECTORY}/lib/libboost_numpy36-mt-d.a" \
  "$1/lib/libboost_numpy36-mt-d.a"

ln \
  "${HUNTER_CELLAR_RAW_DIRECTORY}/lib/libboost_numpy36-mt.a" \
  "$1/lib/libboost_numpy36-mt.a"

ln \
  "${HUNTER_CELLAR_RAW_DIRECTORY}/lib/libboost_python36-mt-d.a" \
  "$1/lib/libboost_python36-mt-d.a"

ln \
  "${HUNTER_CELLAR_RAW_DIRECTORY}/lib/libboost_python36-mt.a" \
  "$1/lib/libboost_python36-mt.a"

