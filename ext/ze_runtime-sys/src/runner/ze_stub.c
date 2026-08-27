#include "../level-zero/ze_api.h"

#define ZE_STUB(name, args) \
    ze_result_t name args { return ZE_RESULT_ERROR_UNINITIALIZED; }

ZE_STUB(zeInit, (uint32_t flags))
ZE_STUB(zeDriverGet, (uint32_t *pCount, ze_driver_handle_t *phDrivers))
ZE_STUB(zeDeviceGet, (ze_driver_handle_t hDriver, uint32_t *pCount,
                      ze_device_handle_t *phDevices))
ZE_STUB(zeContextCreate, (ze_driver_handle_t hDriver,
                          const ze_context_desc_t *desc,
                          ze_context_handle_t *phContext))
ZE_STUB(zeContextDestroy, (ze_context_handle_t hContext))
ZE_STUB(zeCommandListCreateImmediate,
        (ze_context_handle_t hContext, ze_device_handle_t hDevice,
         const ze_command_queue_desc_t *desc,
         ze_command_list_handle_t *phCommandList))
ZE_STUB(zeCommandListDestroy, (ze_command_list_handle_t hCommandList))
ZE_STUB(zeModuleCreate,
        (ze_context_handle_t hContext, ze_device_handle_t hDevice,
         const ze_module_desc_t *desc, ze_module_handle_t *phModule,
         ze_module_build_log_handle_t *phBuildLog))
ZE_STUB(zeModuleDestroy, (ze_module_handle_t hModule))
ZE_STUB(zeModuleBuildLogGetString,
        (ze_module_build_log_handle_t hModuleBuildLog, size_t *pSize,
         char *pBuildLog))
ZE_STUB(zeModuleBuildLogDestroy,
        (ze_module_build_log_handle_t hModuleBuildLog))
ZE_STUB(zeKernelCreate,
        (ze_module_handle_t hModule, const ze_kernel_desc_t *desc,
         ze_kernel_handle_t *phKernel))
ZE_STUB(zeKernelDestroy, (ze_kernel_handle_t hKernel))
ZE_STUB(zeKernelSetGroupSize,
        (ze_kernel_handle_t hKernel, uint32_t groupSizeX,
         uint32_t groupSizeY, uint32_t groupSizeZ))
ZE_STUB(zeKernelSetArgumentValue,
        (ze_kernel_handle_t hKernel, uint32_t argIndex, size_t argSize,
         const void *pArgValue))
ZE_STUB(zeMemAllocDevice,
        (ze_context_handle_t hContext,
         const ze_device_mem_alloc_desc_t *deviceDesc, size_t size,
         size_t alignment, ze_device_handle_t hDevice, void **pptr))
ZE_STUB(zeMemFree, (ze_context_handle_t hContext, void *ptr))
ZE_STUB(zeCommandListAppendMemoryCopy,
        (ze_command_list_handle_t hCommandList, void *dstptr,
         const void *srcptr, size_t size, ze_event_handle_t hSignalEvent,
         uint32_t numWaitEvents, ze_event_handle_t *phWaitEvents))
ZE_STUB(zeCommandListAppendBarrier,
        (ze_command_list_handle_t hCommandList,
         ze_event_handle_t hSignalEvent, uint32_t numWaitEvents,
         ze_event_handle_t *phWaitEvents))
ZE_STUB(zeCommandListAppendLaunchKernel,
        (ze_command_list_handle_t hCommandList, ze_kernel_handle_t hKernel,
         const ze_group_count_t *launchArgs,
         ze_event_handle_t hSignalEvent, uint32_t numWaitEvents,
         ze_event_handle_t *phWaitEvents))
ZE_STUB(zeCommandListHostSynchronize,
        (ze_command_list_handle_t hCommandList, uint64_t timeout))
