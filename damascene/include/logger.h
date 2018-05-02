/**
 * Copyright (c) 2017 rxi
 *
 * This library is free software; you can redistribute it and/or modify it
 * under the terms of the MIT license. See `log.c` for details.
 */

#ifndef LOG_H
#define LOG_H

#include <stdio.h>
#include <stdarg.h>
#include <string.h>

#define LOG_VERSION "0.1.0"

typedef void (*log_LockFn)(void *udata, int lock);

enum { LOG_TRACE, LOG_DEBUG, LOG_INFO, LOG_WARN, LOG_ERROR, LOG_FATAL };

#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

// #define log_trace(...) log_log(LOG_TRACE, __FILENAME__, __LINE__, __VA_ARGS__)
// #define log_debug(...) log_log(LOG_DEBUG, __FILENAME__, __LINE__, __VA_ARGS__)
// #define log_info(...)  log_log(LOG_INFO,  __FILENAME__, __LINE__, __VA_ARGS__)
// #define log_warn(...)  log_log(LOG_WARN,  __FILENAME__, __LINE__, __VA_ARGS__)
// #define log_error(...) log_log(LOG_ERROR, __FILENAME__, __LINE__, __VA_ARGS__)
// #define log_fatal(...) log_log(LOG_FATAL, __FILENAME__, __LINE__, __VA_ARGS__)

#define log_trace(...) do { printf(__VA_ARGS__); printf("\n"); } while (0)
#define log_debug(...) do { printf(__VA_ARGS__); printf("\n"); } while (0)
#define log_info(...)  do { printf(__VA_ARGS__); printf("\n"); } while (0)
#define log_warn(...) do { printf(__VA_ARGS__); printf("\n"); } while (0)
#define log_error(...) do { printf(__VA_ARGS__); printf("\n"); } while (0)
#define log_fatal(...) do { printf(__VA_ARGS__); printf("\n"); } while (0)

extern "C" void log_set_udata(void *udata);
extern "C" void log_set_lock(log_LockFn fn);
extern "C" void log_set_fp(FILE *fp);
extern "C" void log_set_level(int level);
extern "C" void log_set_quiet(int enable);

extern "C" void log_log(int level, const char *file, int line, const char *fmt, ...);

#endif
