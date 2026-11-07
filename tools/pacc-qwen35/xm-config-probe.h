#ifndef HETGPU_XM_CONFIG_PROBE_H
#define HETGPU_XM_CONFIG_PROBE_H
#include <stdint.h>
struct XmConfigProbe {
    uint32_t formats[4][16];
    uint32_t guest[8];
};
void hetgpu_xm_config_probe(struct XmConfigProbe *out);
#endif
