/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */

#ifndef _C_ARITHMETICS_C_IS_EQUAL_H_
#define _C_ARITHMETICS_C_IS_EQUAL_H_


// =======
// Headers
// =======

#include <cmath>  // std::fabs
#include <limits>  // epsilon

// =============
// c arithmetics
// =============

/// \namespace c_arithmetics
///
/// \brief     This namespace declares arithmetic functions in C.
///
/// \sa        c_arithmetics::is_equal

namespace c_arithmetics
{

    // ========
    // is equal
    // ========

    /// \brief     Check if two floating point numbers are equal within a
    ///            tolerance.
    ///
    /// \param[in] x
    ///            A float number
    /// \param[in] y
    ///            A float number
    /// \return    \c true or \c false .

    template <typename DataType>
    inline bool is_equal(DataType x, DataType y)
    {
        if (std::fabs(x - y) <= 2.0 * std::numeric_limits<DataType>::epsilon())
        {
            return true;
        }
        else
        {
            return false;
        }
    }

}  // namepsace c_arithmetics

#endif  // _C_ARITHMETICS_C_IS_EQUAL_H_
