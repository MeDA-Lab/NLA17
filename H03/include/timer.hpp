////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    timer.hpp
/// @brief   The timer header.
///
/// @author  Mu Yang <<emfomy@gmail.com>>
///

#ifndef SCSC_TIMER_HPP
#define SCSC_TIMER_HPP

#include <iostream>
#include <sys/time.h>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief  Gets current time.
///
/// @return  current time.
///
inline double getTime() {
  timeval tv;
  gettimeofday(&tv, NULL);
  return double(tv.tv_sec) + double(tv.tv_usec) * 1e-6;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief  Start stopwatch timer.
///
/// @param  ptr_timer  the timer; pointer.
///
/// @see  toc
///
inline double tic( double *ptr_timer ) {
  *ptr_timer = getTime();
  return *ptr_timer;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief  Read elapsed time from stopwatch
///
/// @param  ptr_timer  the timer; pointer.
///
/// @return  elapsed time.
///
/// @see  tic
///
inline double toc( double *ptr_timer ) {
  double time = getTime() - *ptr_timer;
  std::cout << "Elapsed time is " << time << " seconds." << std::endl;
  return time;
}

#endif  // SCSC_TIMER_HPP
