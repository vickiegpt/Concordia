#include <pthread.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include "q6k-xm-profile.h"

extern int hetgpu_q6k_q8_1_xm_serial(const uint8_t *, const uint8_t *, float *,
                                    size_t, size_t, size_t, size_t, size_t, size_t);

enum { WORKERS = 4 };
struct Work {
    const uint8_t *x, *y;
    float *dst;
    size_t m, n, nb, sx, sy, sd;
    int result;
    struct Q6kXmProfile profile;
};
static struct Work work[WORKERS];
static pthread_t threads[WORKERS];
static pthread_once_t once = PTHREAD_ONCE_INIT;
static pthread_mutex_t dispatch = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t ready = PTHREAD_COND_INITIALIZER, finished = PTHREAD_COND_INITIALIZER;
static uint64_t generation;
static unsigned pending;
static bool stop, initialized;

static void *worker(void *opaque) {
    struct Work *slot = opaque;
    uint64_t seen = 0;
    pthread_mutex_lock(&mutex);
    for (;;) {
        while (!stop && seen == generation) pthread_cond_wait(&ready, &mutex);
        if (stop) break;
        seen = generation;
        struct Work job = *slot;
        pthread_mutex_unlock(&mutex);
        int result = hetgpu_q6k_q8_1_xm_serial(job.x, job.y, job.dst, job.m,
                                             job.n, job.nb, job.sx, job.sy, job.sd);
        struct Q6kXmProfile profile = q6k_xm_profile;
        pthread_mutex_lock(&mutex);
        slot->result = result;
        slot->profile = profile;
        if (!--pending) pthread_cond_signal(&finished);
    }
    pthread_mutex_unlock(&mutex);
    return NULL;
}

static void initialize(void) {
    pthread_attr_t attr;
    if (pthread_attr_init(&attr)) return;
    if (pthread_attr_setstacksize(&attr, 256*1024)) {
        pthread_attr_destroy(&attr);
        return;
    }
    unsigned count;
    for (count = 0; count < WORKERS; ++count)
        if (pthread_create(&threads[count], &attr, worker, &work[count])) break;
    pthread_attr_destroy(&attr);
    if (count == WORKERS) { initialized = true; return; }
    pthread_mutex_lock(&mutex);
    stop = true;
    pthread_cond_broadcast(&ready);
    pthread_mutex_unlock(&mutex);
    for (unsigned i = 0; i < count; ++i) pthread_join(threads[i], NULL);
}

int hetgpu_q6k_q8_1_batch_xm(const uint8_t *x, const uint8_t *y, float *dst,
                           size_t m, size_t n, size_t nb, size_t sx, size_t sy, size_t sd) {
    if (!m || !n || n > 8 || !nb || nb > 128 || sx < nb*210 || sy < nb*288 || sd < m)
        return -1;
    if (m < WORKERS*32)
        return hetgpu_q6k_q8_1_xm_serial(x, y, dst, m, n, nb, sx, sy, sd);
    pthread_mutex_lock(&dispatch);
    pthread_once(&once, initialize);
    if (!initialized) { pthread_mutex_unlock(&dispatch); return -1; }
    pthread_mutex_lock(&mutex);
    size_t start = 0;
    for (unsigned i = 0; i < WORKERS; ++i) {
        size_t rows = m/WORKERS+(i < m%WORKERS);
        work[i] = (struct Work){.x=x+start*sx, .y=y, .dst=dst+start,
            .m=rows, .n=n, .nb=nb, .sx=sx, .sy=sy, .sd=sd};
        start += rows;
    }
    pending = WORKERS;
    ++generation;
    pthread_cond_broadcast(&ready);
    while (pending) pthread_cond_wait(&finished, &mutex);
    int result = 0;
    memset(&q6k_xm_profile, 0, sizeof(q6k_xm_profile));
    for (unsigned i = 0; i < WORKERS; ++i) {
        result |= work[i].result;
        q6k_xm_profile.activation += work[i].profile.activation;
        q6k_xm_profile.weights += work[i].profile.weights;
        q6k_xm_profile.matrix += work[i].profile.matrix;
        q6k_xm_profile.reduce += work[i].profile.reduce;
        q6k_xm_profile.calls += work[i].profile.calls;
        q6k_xm_profile.workers += work[i].profile.workers;
    }
    pthread_mutex_unlock(&mutex);
    pthread_mutex_unlock(&dispatch);
    return result;
}
