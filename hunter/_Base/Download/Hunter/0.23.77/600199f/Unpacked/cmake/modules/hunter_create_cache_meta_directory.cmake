# Copyright (c) 2015, Ruslan Baratov
# All rights reserved.

include(hunter_internal_error)
include(hunter_make_directory)
include(hunter_assert_not_empty_string)

function(hunter_create_cache_meta_directory cache_directory result)
  hunter_assert_not_empty_string("${HUNTER_ARGS_FILE}")
  hunter_assert_not_empty_string("${HUNTER_PACKAGE_CONFIGURATION_TYPES}")
  hunter_assert_not_empty_string("${HUNTER_PACKAGE_HOME_DIR}")
  hunter_assert_not_empty_string("${HUNTER_PACKAGE_NAME}")
  hunter_assert_not_empty_string("${HUNTER_PACKAGE_SHA1}")
  hunter_assert_not_empty_string("${HUNTER_PACKAGE_VERSION}")
  hunter_assert_not_empty_string("${HUNTER_TOOLCHAIN_ID_PATH}")
  hunter_assert_not_empty_string("${HUNTER_TOOLCHAIN_SHA1}")
  hunter_assert_not_empty_string("${cache_directory}")
  hunter_assert_not_empty_string("${result}")

  string(COMPARE NOTEQUAL "${HUNTER_PACKAGE_COMPONENT}" "" has_component)

  # Save toolchain-id
  hunter_make_directory(
      "${cache_directory}/meta" "${HUNTER_TOOLCHAIN_SHA1}" cache_meta_dir
  )
  set(toolchain_info "${HUNTER_TOOLCHAIN_ID_PATH}/toolchain.info")
  if(NOT EXISTS "${toolchain_info}")
    hunter_internal_error("Not exists: ${toolchain_info}")
  endif()
  file(COPY "${toolchain_info}" DESTINATION "${cache_meta_dir}")

  # Save package name and version
  set(cache_meta_dir "${cache_meta_dir}/${HUNTER_PACKAGE_NAME}")
  if(has_component)
    set(cache_meta_dir "${cache_meta_dir}/__${HUNTER_PACKAGE_COMPONENT}")
  endif()
  set(cache_meta_dir "${cache_meta_dir}/${HUNTER_PACKAGE_VERSION}")

  # Save package archive-id
  hunter_make_directory(
      "${cache_meta_dir}" "${HUNTER_PACKAGE_SHA1}" cache_meta_dir
  )

  # Save package args
  if(NOT EXISTS "${HUNTER_ARGS_FILE}")
    hunter_internal_error("Args file missing")
  endif()
  file(SHA1 "${HUNTER_ARGS_FILE}" args_sha1)
  hunter_make_directory("${cache_meta_dir}" "${args_sha1}" cache_meta_dir)
  file(COPY "${HUNTER_ARGS_FILE}" DESTINATION "${cache_meta_dir}")

  # Save package configuration types {

  set(types_info "${HUNTER_PACKAGE_HOME_DIR}/types.info")
  set(types_info_nolf "${types_info}.NOLF")

  file(WRITE "${types_info_nolf}" "${HUNTER_PACKAGE_CONFIGURATION_TYPES}")

  # About '@ONLY': no substitutions expected but COPYONLY can't be
  # used with NEWLINE_STYLE
  configure_file(
      "${types_info_nolf}"
      "${types_info}"
      @ONLY
      NEWLINE_STYLE LF
  )

  file(SHA1 "${types_info}" types_sha1)
  hunter_make_directory("${cache_meta_dir}" "${types_sha1}" cache_meta_dir)
  file(COPY "${types_info}" DESTINATION "${cache_meta_dir}")

  # }

  # Save internal-dependencies information {

  set(internal_deps_id "${HUNTER_PACKAGE_HOME_DIR}/internal_deps.id")
  set(internal_deps_id_nolf "${internal_deps_id}.NOLF")
  file(WRITE "${internal_deps_id_nolf}" "${HUNTER_PACKAGE_INTERNAL_DEPS_ID}")

  # About '@ONLY': no substitutions expected but COPYONLY can't be
  # used with NEWLINE_STYLE
  configure_file(
      "${internal_deps_id_nolf}"
      "${internal_deps_id}"
      @ONLY
      NEWLINE_STYLE LF
  )

  file(SHA1 "${internal_deps_id}" internal_deps_sha1)
  hunter_make_directory(
      "${cache_meta_dir}" "${internal_deps_sha1}" cache_meta_dir
  )
  file(COPY "${internal_deps_id}" DESTINATION "${cache_meta_dir}")

  # }

  set("${result}" "${cache_meta_dir}" PARENT_SCOPE)
endfunction()
