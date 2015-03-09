#include <android/log.h>

#ifndef __log_h__
#define __log_h__


#define LOGI(...) ((void)__android_log_print(ANDROID_LOG_INFO, "native-activity", __VA_ARGS__))
#define LOGW(...) ((void)__android_log_print(ANDROID_LOG_WARN, "native-activity", __VA_ARGS__))


#include <time.h>
#define BILLION  1000000000L

// from android samples
/* return current time in milliseconds */
static double now_s(void) {

    struct timespec res;
    clock_gettime(CLOCK_REALTIME, &res);
    return (res.tv_sec + (double) res.tv_nsec / BILLION);

}

#endif