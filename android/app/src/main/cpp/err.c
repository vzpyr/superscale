#include "err.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

int sc_err(char** err, const char* fmt, ...) {
  if (!err) {
    return 1;
  }
  char buf[512];
  va_list args;
  va_start(args, fmt);
  vsnprintf(buf, sizeof(buf), fmt, args);
  va_end(args);
  *err = strdup(buf);
  return 1;
}
