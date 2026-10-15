#ifndef CUMLRKT_API_H
#define CUMLRKT_API_H

#ifdef __GNUC__
#define CUMLRKT_API __attribute__((visibility("default")))
#else
#define CUMLRKT_API
#endif

#endif
