FILE(GLOB KERNEL_FILES ${KERNEL_SOURCE_DIR}/kernels/*.${KERNEL_FILE_EXTENSION})
SET(KERNEL_FILE_DECLARATIONS)
CONFIGURE_FILE(${KERNEL_SOURCE_DIR}/${KERNEL_SOURCE_CLASS}.cpp.in ${KERNELS_CPP})
# Determine file extension length
STRING(LENGTH ${KERNEL_FILE_EXTENSION} extension_length)
# add one space for the dot
MATH(EXPR extension_length ${extension_length}+1)
FOREACH(file ${KERNEL_FILES})
    # Load the file contents and process it.
    FILE(STRINGS ${file} file_content NEWLINE_CONSUME)
    # Replace all backslashes by double backslashes as they are being put in a C string.
    # Be careful not to replace the backslash before a semicolon as that is the CMAKE
    # internal escaping of a semicolon to prevent it from acting as a list seperator.
    STRING(REGEX REPLACE "\\\\([^;])" "\\\\\\\\\\1" file_content "${file_content}")
    # Escape double quotes as being put in a C string.
    STRING(REPLACE "\"" "\\\"" file_content "${file_content}")
    # Split in separate C strings for each line.
    STRING(REPLACE "\n" "\\n\"\n\"" file_content "${file_content}")

    # Determine a name for the variable that will contain this file's contents
    FILE(RELATIVE_PATH filename ${KERNEL_SOURCE_DIR}/kernels ${file})
    STRING(LENGTH ${filename} filename_length)
    MATH(EXPR filename_length ${filename_length}-${extension_length})
    STRING(SUBSTRING ${filename} 0 ${filename_length} variable_name)

    # Record the variable declaration and definition.
    SET(KERNEL_FILE_DECLARATIONS ${KERNEL_FILE_DECLARATIONS}static\ const\ std::string\ ${variable_name};\n)
    FILE(APPEND ${KERNELS_CPP} const\ string\ ${KERNEL_SOURCE_CLASS}::${variable_name}\ =\ \"${file_content}\"\;\n)
ENDFOREACH(file)
CONFIGURE_FILE(${KERNEL_SOURCE_DIR}/${KERNEL_SOURCE_CLASS}.h.in ${KERNELS_H})
