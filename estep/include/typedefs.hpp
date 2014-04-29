/*****************************************************************/
/*    Copyright (c) 2014, Stanford University and the Authors    */
/*    Author: Robert McGibbon <rmcgibbo@gmail.com>               */
/*    Contributors:                                              */
/*                                                               */
/*****************************************************************/

#ifndef MIXTAPE_TYPEDEFS_H
#define MIXTAPE_TYPEDEFS_H
#include <iostream>
#include <boost/multi_array.hpp>

namespace Mixtape {

typedef boost::const_multi_array_ref<float, 2> ConstFloatArray2D;
typedef boost::multi_array<float, 2> FloatArray2D;
typedef boost::multi_array<float, 1> FloatArray1D;
typedef boost::multi_array<double, 2> DoubleArray2D;
typedef boost::multi_array<double, 1> DoubleArray1D;

} // namespace
#endif
