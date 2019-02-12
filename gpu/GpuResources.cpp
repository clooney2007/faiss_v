/*
 * Author: VincentLee
 * Email:  lichlee@yeah.net
 * Created on 2019/01/07.
*/

#include "GpuResources.h"
#include "utils/DeviceUtils.h"

namespace faiss_v { namespace  gpu {

GpuResources::~GpuResources() {

}

DeviceMemory&
GpuResources::getMemoryManagerCurrentDevice() {
    return getMemoryManager(getCurrentDevice());
}

}}