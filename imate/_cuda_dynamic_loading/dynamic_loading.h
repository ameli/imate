/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


#ifndef _CUDA_DYNAMIC_LOADING_DYNAMIC_LOADING_H_
#define _CUDA_DYNAMIC_LOADING_DYNAMIC_LOADING_H_


// =======
// Headers
// =======

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && \
    !defined(__CYGWIN__)
    #include <windows.h>  // HMODULE, TEXT, LoadLibrary, GetProcAddress
#elif defined(__unix__) || defined(__unix) || \
    (defined(__APPLE__) && defined(__MACH__))
    #include <dlfcn.h>  // dlopen, dlsym
#else
    #error "Unknown compiler"
#endif
#include <stdexcept>  // std::runtime_error
#include <sstream>  // std::ostringstream


// ===============
// dynamic loading
// ===============

/// \namespace dynamic_loading
///
/// \brief     Dynamic loading of shared libraries using \c dlopen tool.
///
/// \details   The dynamic loading is compiled with \c CUDA_DYNAMIC_LOADING set
///            to \c 1. When enabeld, the CUDA libraries are loaded at the run-
///            time. This method has an advantage and a disadvatage:
///
///            * Advantage: when dynamic loading is enabled, this package can
///              be distributed withoput bundling the large cuda libraris.
///              That is, when the wheel of this package is repaired with
///              \c auditwheel tool in manylinux platform, the cuda libraries
///              will not be bundled to the wheel, so the size of the wheel
///              does not increase. In contrast, if this package is not
///              compiled with dynamic loading, the cuda \c *.so (or \c *.dll)
///              shared library files will be bundled with the wheel and
///              increase the size of the wheel upto 400MB. Such a large wheel
///              file cannot be uploaded to PyPi due to the 100MB size limit.
///              But with dynamic loading, the package requires these libraries
///              only at the run-time, not at the compile time.
///            * Disadvantage: The end-user must install the *same* cuda
///              version that this package is compiled with. For example, if
///              this package is compiled with cuda 11.x, the user must install
///              cuda 11.x (such as 11.0, 11.1, etc), but not cuda 10.x.
///
///            The run time with/withput dynamic loading is the same. That is,
///            using the dynamic loading doesn't affect the performance at all.
///
///            The functions in this namespace load any generic libraries.
///            We use them to load \c libcudart, \c libcublas, and \c
///            libcusparse.
///
/// \note      When this package is compiled with dynamic loading enabled, make
///            sure that cuda toolkit is available at run-time. For instance
///            on a linux cluster, run:
///
///                module load cuda
///
/// \s         cudartSymbols,
///            cublasSymbols,
///            cusparseSymbols

namespace dynamic_loading
{
    // ==================
    // get library handle (unix)
    // ==================

    #if defined(__unix__) || defined(__unix) || \
        (defined(__APPLE__) && defined(__MACH__))
        /// \brief     Loads a library and returns its handle. This function is
        ///            compiled only on unix-like compiler.
        ///
        /// \details   This function is declared as \c static, which means it
        ///            is only available withing this namespace.
        ///
        /// \param[in] lib_name
        ///            Name of the library.
        /// \return    A pointer to the handle of the loaded library.

        static void* get_library_handle_unix(const char* lib_name)
        { 
            void* handle = dlopen(lib_name, RTLD_LAZY);

            if (!handle)
            {
                throw std::runtime_error(dlerror());
            }

            return handle;
        }
    #endif


    // ==================
    // get library handle (windows)
    // ==================

    #if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && \
            !defined(__CYGWIN__)
        /// \brief     Loads a library and returns its handle. This function is
        ///            compiled only on a windows compiler.
        ///
        /// \details   This function is declared as \c static, which means it
        ///            is only available withing this namespace.
        ///
        /// \param[in] lib_name
        ///            Name of the library.
        /// \return    A pointer to the handle of the loaded library.

        static HMODULE get_library_handle_windows(const char* lib_name)
        { 
            HMODULE handle = LoadLibrary(TEXT(lib_name));

            if (!handle)
            {
                std::ostringstream oss;
                oss << "Cannot load the shared library '" << lib_name << "'." \
                    << std::endl;
                std::string message = oss.str();
                throw std::runtime_error(message);
            }

            return handle;
        }
    #endif


    // ===========
    // load symbol
    // ===========

    /// \brief     Loads a symbol within a library and returns a pointer to
    ///            the symbol (function pointer).
    ///
    /// \tparam    Signature
    ///            The template parameter. The returned symbol pointer is cast
    ///            from \c void* to the template parameter \c Signature, which
    ///            is essentially a \c typedef for the output function.
    /// \param[in] lib_name
    ///            Name of the library.
    /// \param[in] symbol_name
    ///            Name of the symbol within the library.
    /// \return    Returns a pointer to the symbol, which is a pointer to a
    ///            callable function.

    template <typename Signature>
    Signature load_symbol(
            const char* lib_name,
            const char* symbol_name)
    {
        #if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && \
                !defined(__CYGWIN__)

            HMODULE handle = get_library_handle_windows(lib_name);
            FARPROC symbol = GetProcAddress(handle, symbol_name);

            if (symbol == NULL)
            {
                std::ostringstream oss;
                oss << "The symbol '" << symbol << "' is failed to load " \
                    << "from the shared library '" << lib_name << "'." \
                    << std::endl;
                std::string message = oss.str();
                throw std::runtime_error(message);
            }

        #elif defined(__unix__) || defined(__unix) || \
            (defined(__APPLE__) && defined(__MACH__))

            void* handle = get_library_handle_unix(lib_name);
            void* symbol = dlsym(handle, symbol_name);

            char *error = dlerror();
            if (error != NULL)
            {
                throw std::runtime_error(dlerror());
            }

        #else
            #error "Unknown compiler"
        #endif


        return reinterpret_cast<Signature>(symbol);
    }

}  // namespace dynamic_loading

#endif  // _CUDA_DYNAMIC_LOADING_DYNAMIC_LOADING_H_
