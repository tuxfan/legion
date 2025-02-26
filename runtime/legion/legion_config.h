/* Copyright 2019 Stanford University, NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#ifndef __LEGION_CONFIG_H__
#define __LEGION_CONFIG_H__

#ifdef LEGION_USE_CMAKE
#include "legion/legion_defines.h"
#endif

#ifndef LEGION_USE_PYTHON_CFFI
// for UINT_MAX, INT_MAX, INT_MIN
#include <limits.h>
#endif // LEGION_USE_PYTHON_CFFI

/**
 * \file legion_config.h
 */

// ******************** IMPORTANT **************************
//
// This file is PURE C, **NOT** C++. Keep any C++-isms in
// legion_types.h, or elsewhere.
//
// ******************** IMPORTANT **************************

#include "realm/realm_c.h"

//==========================================================================
//                                Constants
//==========================================================================

#ifndef AUTO_GENERATE_ID
#define AUTO_GENERATE_ID   UINT_MAX
#else
#error "legion.h requires the ability to define the macro 'AUTO_GENERATE_ID' but it has already been defined"
#endif

#ifndef GC_MIN_PRIORITY
#define GC_MIN_PRIORITY    INT_MIN
#else
#error "legion.h requires the ability to define the macro 'GC_MIN_PRIORITY' but it has already been defined"
#endif

#ifndef GC_MAX_PRIORITY
#define GC_MAX_PRIORITY    INT_MAX
#else
#error "legion.h requires the ability to define the macro 'GC_MAX_PRIORITY' but it has already been defined"
#endif

#ifndef GC_NEVER_PRIORITY
#define GC_NEVER_PRIORITY  GC_MIN_PRIORITY
#else
#error "legion.h requires the ability to define the macro 'GC_NEVER_PRIORITY' but it has already been defined"
#endif

#ifndef GC_FIRST_PRIORITY
#define GC_FIRST_PRIORITY  GC_MAX_PRIORITY
#endif
#ifndef GC_DEFAULT_PRIORITY
#define GC_DEFAULT_PRIORITY 0
#endif
#ifndef GC_LAST_PRIORITY
#define GC_LAST_PRIORITY   (GC_MIN_PRIORITY+1)
#endif

#ifndef LEGION_MAX_DIM
#define LEGION_MAX_DIM     3 // maximum number of dimensions for index spaces
#endif

#ifndef MAX_RETURN_SIZE // For backwards compatibility
#ifndef LEGION_MAX_RETURN_SIZE
#define LEGION_MAX_RETURN_SIZE    2048 // maximum return type size in bytes
#endif
#else
#ifndef LEGION_MAX_RETURN_SIZE
#define LEGION_MAX_RETURN_SIZE    (MAX_RETURN_SIZE)
#endif
#endif

#ifndef MAX_FIELDS // For backwards compatibility
#ifndef LEGION_MAX_FIELDS
#define LEGION_MAX_FIELDS         512 // must be a power of 2
#endif
#else
#ifndef LEGION_MAX_FIELDS
#define LEGION_MAX_FIELDS         (MAX_FIELDS)
#endif
#endif

// Some default values

// The maximum number of nodes to be run on
#ifndef MAX_NUM_NODES // For backwards compatibility
#ifndef LEGION_MAX_NUM_NODES
#define LEGION_MAX_NUM_NODES                   1024 // must be a power of 2
#endif
#else
#ifndef LEGION_MAX_NUM_NODES
#define LEGION_MAX_NUM_NODES                   (MAX_NUM_NODES)
#endif
#endif
// The maximum number of processors on a node
#ifndef MAX_NUM_PROCS // For backwards compatibility
#ifndef LEGION_MAX_NUM_PROCS
#define LEGION_MAX_NUM_PROCS                   64 // must be a power of 2
#endif
#else
#ifndef LEGION_MAX_NUM_PROCS
#define LEGION_MAX_NUM_PROCS                   (MAX_NUM_PROCS)
#endif
#endif
// Maximum ID for an application task ID 
#ifndef MAX_APPLICATION_TASK_ID // For backwards compatibility
#ifndef LEGION_MAX_APPLICATION_TASK_ID
#define LEGION_MAX_APPLICATION_TASK_ID         (1<<20)
#endif
#else
#ifndef LEGION_MAX_APPLICATION_TASK_ID
#define LEGION_MAX_APPLICATION_TASK_ID         (MAX_APPLICATION_TASK_ID)
#endif
#endif
// Maximum ID for an application field ID
#ifndef MAX_APPLICATION_FIELD_ID // For backwards compatibility
#ifndef LEGION_MAX_APPLICATION_FIELD_ID
#define LEGION_MAX_APPLICATION_FIELD_ID        (1<<20)
#endif
#else
#ifndef LEGION_MAX_APPLICATION_FIELD_ID
#define LEGION_MAX_APPLICATION_FIELD_ID        (MAX_APPLICATION_FIELD_ID)
#endif
#endif
// Maximum ID for an application mapper ID
#ifndef MAX_APPLICATION_MAPPER_ID // For backwards compatibility
#ifndef LEGION_MAX_APPLICATION_MAPPER_ID
#define LEGION_MAX_APPLICATION_MAPPER_ID       (1<<20)
#endif
#else
#ifndef LEGION_MAX_APPLICATION_MAPPER_ID
#define LEGION_MAX_APPLICATION_MAPPER_ID       (MAX_APPLICATION_MAPPER_ID)
#endif
#endif
// Maximum ID for an application projection ID
#ifndef MAX_APPLICATION_PROJECTION_ID // For backwards compatibility
#ifndef LEGION_MAX_APPLICATION_PROJECTION_ID
#define LEGION_MAX_APPLICATION_PROJECTION_ID   (1<<20)
#endif
#else
#ifndef LEGION_MAX_APPLICATION_PROJECTION_ID
#define LEGION_MAX_APPLICATION_PROJECTION_ID  (MAX_APPLICATION_PROJECTION_ID)
#endif
#endif
// Maximum ID for an application sharding ID
#ifndef MAX_APPLICATION_SHARDING_ID // For backwards compatibility
#ifndef LEGION_MAX_APPLICATION_SHARDING_ID
#define LEGION_MAX_APPLICATION_SHARDING_ID     (1<<20)
#endif
#else
#ifndef LEGION_MAX_APPLICATION_SHARDING_ID
#define LEGION_MAX_APPLICATION_SHARDING_ID    (MAX_APPLICATION_SHARDING_ID)
#endif
#endif
// Maximum ID for an application reduction ID
#ifndef LEGION_MAX_APPLICATION_REDOP_ID
#define LEGION_MAX_APPLICATION_REDOP_ID       (1<<20)
#endif
// Maximum ID for an application serdez ID
#ifndef LEGION_MAX_APPLICATION_SERDEZ_ID
#define LEGION_MAX_APPLICATION_SERDEZ_ID      (1<<20)
#endif
// Default number of local fields per field space
#ifndef DEFAULT_LOCAL_FIELDS // For backwards compatibility
#ifndef LEGION_DEFAULT_LOCAL_FIELDS
#define LEGION_DEFAULT_LOCAL_FIELDS            4
#endif
#else
#ifndef LEGION_DEFAULT_LOCAL_FIELDS
#define LEGION_DEFAULT_LOCAL_FIELDS            (DEFAULT_LOCAL_FIELDS)
#endif
#endif
// Default number of mapper slots
#ifndef DEFAULT_MAPPER_SLOTS // For backwards compatibility
#ifndef LEGION_DEFAULT_MAPPER_SLOTS
#define LEGION_DEFAULT_MAPPER_SLOTS            8
#endif
#else
#ifndef LEGION_DEFAULT_MAPPER_SLOTS
#define LEGION_DEFAULT_MAPPER_SLOTS            (DEFAULT_MAPPER_SLOTS)
#endif
#endif
// Default number of contexts made for each runtime instance
// Ideally this is a power of 2 (better for performance)
#ifndef DEFAULT_CONTEXTS // For backwards compatibility
#ifndef LEGION_DEFAULT_CONTEXTS
#define LEGION_DEFAULT_CONTEXTS                8
#endif
#else
#ifndef LEGION_DEFAULT_CONTEXTS
#define LEGION_DEFAULT_CONTEXTS                (DEFAULT_CONTEXTS)
#endif
#endif
// Maximum number of sub-tasks per task at a time
#ifndef DEFAULT_MAX_TASK_WINDOW // For backwards compatibility
#ifndef LEGION_DEFAULT_MAX_TASK_WINDOW
#define LEGION_DEFAULT_MAX_TASK_WINDOW         1024
#endif
#else
#ifndef LEGION_DEFAULT_MAX_TASK_WINDOW
#define LEGION_DEFAULT_MAX_TASK_WINDOW         (DEFAULT_MAX_TASK_WINDOW)
#endif
#endif
// Default amount of hysteresis on the task window in the
// form of a percentage (must be between 0 and 100)
#ifndef DEFAULT_TASK_WINDOW_HYSTERESIS // For backwards compatibility
#ifndef LEGION_DEFAULT_TASK_WINDOW_HYSTERESIS
#define LEGION_DEFAULT_TASK_WINDOW_HYSTERESIS  25
#endif
#else
#ifndef LEGION_DEFAULT_TASK_WINDOW_HYSTERESIS
#define LEGION_DEFAULT_TASK_WINDOW_HYSTERESIS  (DEFAULT_TASK_WINDOW_HYSTERESIS)
#endif
#endif
// Default number of tasks to have in flight before applying 
// back pressure to the mapping process for a context
#ifndef DEFAULT_MIN_TASKS_TO_SCHEDULE // For backwards compatibility
#ifndef LEGION_DEFAULT_MIN_TASKS_TO_SCHEDULE
#define LEGION_DEFAULT_MIN_TASKS_TO_SCHEDULE   32
#endif
#else
#ifndef LEGION_DEFAULT_MIN_TASKS_TO_SCHEDULE
#define LEGION_DEFAULT_MIN_TASKS_TO_SCHEDULE   (DEFAULT_MIN_TASKS_TO_SCHEDULE)
#endif
#endif
// How many tasks to group together for runtime operations
#ifndef DEFAULT_META_TASK_VECTOR_WIDTH // For backwards compatibility
#ifndef LEGION_DEFAULT_META_TASK_VECTOR_WIDTH
#define LEGION_DEFAULT_META_TASK_VECTOR_WIDTH  16
#endif
#else
#ifndef LEGION_DEFAULT_META_TASK_VECTOR_WIDTH
#define LEGION_DEFAULT_META_TASK_VECTOR_WIDTH  (DEFAULT_META_TASK_VECTOR_WIDTH)
#endif
#endif
// Default number of replay tasks to run in parallel
#ifndef DEFAULT_MAX_REPLAY_PARALLELISM // For backwards compatibility
#ifndef LEGION_DEFAULT_MAX_REPLAY_PARALLELISM
#define LEGION_DEFAULT_MAX_REPLAY_PARALLELISM  2
#endif
#else
#ifndef LEGION_DEFAULT_MAX_REPLAY_PARALLELISM
#define LEGION_DEFAULT_MAX_REPLAY_PARALLELISM  (DEFAULT_MAX_REPLAY_PARALLELISM)
#endif
#endif
// The maximum size of active messages sent by the runtime in bytes
// Note this value was picked based on making a tradeoff between
// latency and bandwidth numbers on both Cray and Infiniband
// interconnect networks.
#ifndef DEFAULT_MAX_MESSAGE_SIZE // For backwards compatibility
#ifndef LEGION_DEFAULT_MAX_MESSAGE_SIZE
#define LEGION_DEFAULT_MAX_MESSAGE_SIZE        16384
#endif
#else
#ifndef LEGION_DEFAULT_MAX_MESSAGE_SIZE
#define LEGION_DEFAULT_MAX_MESSAGE_SIZE        (DEFAULT_MAX_MESSAGE_SIZE)
#endif
#endif
// Timeout before checking for whether a logical user
// should be pruned from the logical region tree data strucutre
// Making the value less than or equal to zero will
// result in checks always being performed
#ifndef DEFAULT_LOGICAL_USER_TIMEOUT // For backwards compatibility
#ifndef LEGION_DEFAULT_LOGICAL_USER_TIMEOUT
#define LEGION_DEFAULT_LOGICAL_USER_TIMEOUT    32
#endif
#else
#ifndef LEGION_DEFAULT_LOGICAL_USER_TIMEOUT
#define LEGION_DEFAULT_LOGICAL_USER_TIMEOUT    (DEFAULT_LOGICAL_USER_TIMEOUT)
#endif
#endif
// Number of events to place in each GC epoch
// Large counts improve efficiency but add latency to
// garbage collection.  Smaller count reduce efficiency
// but improve latency of collection.
#ifndef DEFAULT_GC_EPOCH_SIZE // For backwards compatibility
#ifndef LEGION_DEFAULT_GC_EPOCH_SIZE
#define LEGION_DEFAULT_GC_EPOCH_SIZE           64
#endif
#else
#ifndef LEGION_DEFAULT_GC_EPOCH_SIZE
#define LEGION_DEFAULT_GC_EPOCH_SIZE           (DEFAULT_GC_EPOCH_SIZE)
#endif
#endif

// Used for debugging memory leaks
// How often tracing information is dumped
// based on the number of scheduler invocations
#ifndef TRACE_ALLOCATION_FREQUENCY // For backwards compatibility
#ifndef LEGION_TRACE_ALLOCATION_FREQUENCY
#define LEGION_TRACE_ALLOCATION_FREQUENCY      1024
#endif
#else
#ifndef LEGION_TRACE_ALLOCATION_FREQUENCY
#define LEGION_TRACE_ALLOCATION_FREQUENCY      (TRACE_ALLOCATION_FREQUENCY)
#endif
#endif

// The maximum alignment guaranteed on the 
// target machine bytes.  For most 64-bit 
// systems this should be 16 bytes.
#ifndef LEGION_MAX_ALIGNMENT
#define LEGION_MAX_ALIGNMENT            16
#endif

// Give an ideal upper bound on the maximum
// number of operations Legion should keep
// available for recycling. Where possible
// the runtime will delete objects to keep
// overall memory usage down.
#ifndef LEGION_MAX_RECYCLABLE_OBJECTS
#define LEGION_MAX_RECYCLABLE_OBJECTS      1024
#endif

// An initial seed for random numbers
// generated by the high-level runtime.
#ifndef LEGION_INIT_SEED
#define LEGION_INIT_SEED                  0x221B
#endif

// The radix for the runtime to use when 
// performing collective operations internally
#ifndef LEGION_COLLECTIVE_RADIX
#define LEGION_COLLECTIVE_RADIX           8
#endif

// The radix for the broadcast tree
// when attempting to shutdown the runtime
#ifndef LEGION_SHUTDOWN_RADIX
#define LEGION_SHUTDOWN_RADIX             8
#endif

// Maximum depth of composite instances before warnings
#ifndef LEGION_PRUNE_DEPTH_WARNING
#define LEGION_PRUNE_DEPTH_WARNING        8
#endif

// Maximum number of non-replayable templates before warnings
#ifndef LEGION_NON_REPLAYABLE_WARNING
#define LEGION_NON_REPLAYABLE_WARNING     5
#endif

// Initial offset for library IDs
// Controls how many IDs are available for dynamic use
#ifndef LEGION_INITIAL_LIBRARY_ID_OFFSET
#define LEGION_INITIAL_LIBRARY_ID_OFFSET (1 << 30)
#endif

// Some helper macros

// This statically computes an integer log base 2 for a number
// which is guaranteed to be a power of 2. Adapted from
// http://graphics.stanford.edu/~seander/bithacks.html#IntegerLogDeBruijn
#define STATIC_LOG2(x)  (LOG2_LOOKUP(uint32_t(x * 0x077CB531U) >> 27))
#define LOG2_LOOKUP(x) ((x==0) ? 0 : (x==1) ? 1 : (x==2) ? 28 : (x==3) ? 2 : \
                   (x==4) ? 29 : (x==5) ? 14 : (x==6) ? 24 : (x==7) ? 3 : \
                   (x==8) ? 30 : (x==9) ? 22 : (x==10) ? 20 : (x==11) ? 15 : \
                   (x==12) ? 25 : (x==13) ? 17 : (x==14) ? 4 : (x==15) ? 8 : \
                   (x==16) ? 31 : (x==17) ? 27 : (x==18) ? 13 : (x==19) ? 23 : \
                   (x==20) ? 21 : (x==21) ? 19 : (x==22) ? 16 : (x==23) ? 7 : \
                   (x==24) ? 26 : (x==25) ? 12 : (x==26) ? 18 : (x==27) ? 6 : \
                   (x==28) ? 11 : (x==29) ? 5 : (x==30) ? 10 : 9)

// log2(LEGION_MAX_FIELDS)
#ifndef LEGION_FIELD_LOG2
#define LEGION_FIELD_LOG2         STATIC_LOG2(LEGION_MAX_FIELDS) 
#endif

#define LEGION_STRINGIFY(x) #x
#define LEGION_MACRO_TO_STRING(x) LEGION_STRINGIFY(x)

#define LEGION_DISTRIBUTED_ID_MASK    0x00FFFFFFFFFFFFFFULL
#define LEGION_DISTRIBUTED_ID_FILTER(x) ((x) & 0x00FFFFFFFFFFFFFFULL)
#define LEGION_DISTRIBUTED_HELP_DECODE(x)   ((x) >> 56)
#define LEGION_DISTRIBUTED_HELP_ENCODE(x,y) ((x) | (((long long)(y)) << 56))

#if LEGION_MAX_DIM == 1

#define LEGION_FOREACH_N(__func__) \
  __func__(1) 
#define LEGION_FOREACH_NN(__func__) \
  __func__(1,1)

#elif LEGION_MAX_DIM == 2

#define LEGION_FOREACH_N(__func__) \
  __func__(1) \
  __func__(2)
#define LEGION_FOREACH_NN(__func__) \
  __func__(1,1) \
  __func__(1,2) \
  __func__(2,1) \
  __func__(2,2)

#elif LEGION_MAX_DIM == 3

#define LEGION_FOREACH_N(__func__) \
  __func__(1) \
  __func__(2) \
  __func__(3)
#define LEGION_FOREACH_NN(__func__) \
  __func__(1,1) \
  __func__(1,2) \
  __func__(1,3) \
  __func__(2,1) \
  __func__(2,2) \
  __func__(2,3) \
  __func__(3,1) \
  __func__(3,2) \
  __func__(3,3)

#elif LEGION_MAX_DIM == 4

#define LEGION_FOREACH_N(__func__) \
  __func__(1) \
  __func__(2) \
  __func__(3) \
  __func__(4)
#define LEGION_FOREACH_NN(__func__) \
  __func__(1,1) \
  __func__(1,2) \
  __func__(1,3) \
  __func__(1,4) \
  __func__(2,1) \
  __func__(2,2) \
  __func__(2,3) \
  __func__(2,4) \
  __func__(3,1) \
  __func__(3,2) \
  __func__(3,3) \
  __func__(3,4) \
  __func__(4,1) \
  __func__(4,2) \
  __func__(4,3) \
  __func__(4,4)

#elif LEGION_MAX_DIM == 5

#define LEGION_FOREACH_N(__func__) \
  __func__(1) \
  __func__(2) \
  __func__(3) \
  __func__(4) \
  __func__(5)
#define LEGION_FOREACH_NN(__func__) \
  __func__(1,1) \
  __func__(1,2) \
  __func__(1,3) \
  __func__(1,4) \
  __func__(1,5) \
  __func__(2,1) \
  __func__(2,2) \
  __func__(2,3) \
  __func__(2,4) \
  __func__(2,5) \
  __func__(3,1) \
  __func__(3,2) \
  __func__(3,3) \
  __func__(3,4) \
  __func__(3,5) \
  __func__(4,1) \
  __func__(4,2) \
  __func__(4,3) \
  __func__(4,4) \
  __func__(4,5) \
  __func__(5,1) \
  __func__(5,2) \
  __func__(5,3) \
  __func__(5,4) \
  __func__(5,5)

#elif LEGION_MAX_DIM == 6

#define LEGION_FOREACH_N(__func__) \
  __func__(1) \
  __func__(2) \
  __func__(3) \
  __func__(4) \
  __func__(5) \
  __func__(6)
#define LEGION_FOREACH_NN(__func__) \
  __func__(1,1) \
  __func__(1,2) \
  __func__(1,3) \
  __func__(1,4) \
  __func__(1,5) \
  __func__(1,6) \
  __func__(2,1) \
  __func__(2,2) \
  __func__(2,3) \
  __func__(2,4) \
  __func__(2,5) \
  __func__(2,6) \
  __func__(3,1) \
  __func__(3,2) \
  __func__(3,3) \
  __func__(3,4) \
  __func__(3,5) \
  __func__(3,6) \
  __func__(4,1) \
  __func__(4,2) \
  __func__(4,3) \
  __func__(4,4) \
  __func__(4,5) \
  __func__(4,6) \
  __func__(5,1) \
  __func__(5,2) \
  __func__(5,3) \
  __func__(5,4) \
  __func__(5,5) \
  __func__(5,6) \
  __func__(6,1) \
  __func__(6,2) \
  __func__(6,3) \
  __func__(6,4) \
  __func__(6,5) \
  __func__(6,6)

#elif LEGION_MAX_DIM == 7

#define LEGION_FOREACH_N(__func__) \
  __func__(1) \
  __func__(2) \
  __func__(3) \
  __func__(4) \
  __func__(5) \
  __func__(6) \
  __func__(7)
#define LEGION_FOREACH_NN(__func__) \
  __func__(1,1) \
  __func__(1,2) \
  __func__(1,3) \
  __func__(1,4) \
  __func__(1,5) \
  __func__(1,6) \
  __func__(1,7) \
  __func__(2,1) \
  __func__(2,2) \
  __func__(2,3) \
  __func__(2,4) \
  __func__(2,5) \
  __func__(2,6) \
  __func__(2,7) \
  __func__(3,1) \
  __func__(3,2) \
  __func__(3,3) \
  __func__(3,4) \
  __func__(3,5) \
  __func__(3,6) \
  __func__(3,7) \
  __func__(4,1) \
  __func__(4,2) \
  __func__(4,3) \
  __func__(4,4) \
  __func__(4,5) \
  __func__(4,6) \
  __func__(4,7) \
  __func__(5,1) \
  __func__(5,2) \
  __func__(5,3) \
  __func__(5,4) \
  __func__(5,5) \
  __func__(5,6) \
  __func__(5,7) \
  __func__(6,1) \
  __func__(6,2) \
  __func__(6,3) \
  __func__(6,4) \
  __func__(6,5) \
  __func__(6,6) \
  __func__(6,7) \
  __func__(7,1) \
  __func__(7,2) \
  __func__(7,3) \
  __func__(7,4) \
  __func__(7,5) \
  __func__(7,6) \
  __func__(7,7)

#elif LEGION_MAX_DIM == 8

#define LEGION_FOREACH_N(__func__) \
  __func__(1) \
  __func__(2) \
  __func__(3) \
  __func__(4) \
  __func__(5) \
  __func__(6) \
  __func__(7) \
  __func__(8)
#define LEGION_FOREACH_NN(__func__) \
  __func__(1,1) \
  __func__(1,2) \
  __func__(1,3) \
  __func__(1,4) \
  __func__(1,5) \
  __func__(1,6) \
  __func__(1,7) \
  __func__(1,8) \
  __func__(2,1) \
  __func__(2,2) \
  __func__(2,3) \
  __func__(2,4) \
  __func__(2,5) \
  __func__(2,6) \
  __func__(2,7) \
  __func__(2,8) \
  __func__(3,1) \
  __func__(3,2) \
  __func__(3,3) \
  __func__(3,4) \
  __func__(3,5) \
  __func__(3,6) \
  __func__(3,7) \
  __func__(3,8) \
  __func__(4,1) \
  __func__(4,2) \
  __func__(4,3) \
  __func__(4,4) \
  __func__(4,5) \
  __func__(4,6) \
  __func__(4,7) \
  __func__(4,8) \
  __func__(5,1) \
  __func__(5,2) \
  __func__(5,3) \
  __func__(5,4) \
  __func__(5,5) \
  __func__(5,6) \
  __func__(5,7) \
  __func__(5,8) \
  __func__(6,1) \
  __func__(6,2) \
  __func__(6,3) \
  __func__(6,4) \
  __func__(6,5) \
  __func__(6,6) \
  __func__(6,7) \
  __func__(6,8) \
  __func__(7,1) \
  __func__(7,2) \
  __func__(7,3) \
  __func__(7,4) \
  __func__(7,5) \
  __func__(7,6) \
  __func__(7,7) \
  __func__(7,8) \
  __func__(8,1) \
  __func__(8,2) \
  __func__(8,3) \
  __func__(8,4) \
  __func__(8,5) \
  __func__(8,6) \
  __func__(8,7) \
  __func__(8,8)

#elif LEGION_MAX_DIM == 9

#define LEGION_FOREACH_N(__func__) \
  __func__(1) \
  __func__(2) \
  __func__(3) \
  __func__(4) \
  __func__(5) \
  __func__(6) \
  __func__(7) \
  __func__(8) \
  __func__(9)
#define LEGION_FOREACH_NN(__func__) \
  __func__(1,1) \
  __func__(1,2) \
  __func__(1,3) \
  __func__(1,4) \
  __func__(1,5) \
  __func__(1,6) \
  __func__(1,7) \
  __func__(1,8) \
  __func__(1,9) \
  __func__(2,1) \
  __func__(2,2) \
  __func__(2,3) \
  __func__(2,4) \
  __func__(2,5) \
  __func__(2,6) \
  __func__(2,7) \
  __func__(2,8) \
  __func__(2,9) \
  __func__(3,1) \
  __func__(3,2) \
  __func__(3,3) \
  __func__(3,4) \
  __func__(3,5) \
  __func__(3,6) \
  __func__(3,7) \
  __func__(3,8) \
  __func__(3,9) \
  __func__(4,1) \
  __func__(4,2) \
  __func__(4,3) \
  __func__(4,4) \
  __func__(4,5) \
  __func__(4,6) \
  __func__(4,7) \
  __func__(4,8) \
  __func__(4,9) \
  __func__(5,1) \
  __func__(5,2) \
  __func__(5,3) \
  __func__(5,4) \
  __func__(5,5) \
  __func__(5,6) \
  __func__(5,7) \
  __func__(5,8) \
  __func__(5,9) \
  __func__(6,1) \
  __func__(6,2) \
  __func__(6,3) \
  __func__(6,4) \
  __func__(6,5) \
  __func__(6,6) \
  __func__(6,7) \
  __func__(6,8) \
  __func__(6,9) \
  __func__(7,1) \
  __func__(7,2) \
  __func__(7,3) \
  __func__(7,4) \
  __func__(7,5) \
  __func__(7,6) \
  __func__(7,7) \
  __func__(7,8) \
  __func__(7,9) \
  __func__(8,1) \
  __func__(8,2) \
  __func__(8,3) \
  __func__(8,4) \
  __func__(8,5) \
  __func__(8,6) \
  __func__(8,7) \
  __func__(8,8) \
  __func__(8,9) \
  __func__(9,1) \
  __func__(9,2) \
  __func__(9,3) \
  __func__(9,4) \
  __func__(9,5) \
  __func__(9,6) \
  __func__(9,7) \
  __func__(9,8) \
  __func__(9,9)

#else
#error "Unsupported LEGION_MAX_DIM"
#endif

// The following enums are all re-exported by
// namespace Legion. These versions are here to facilitate the
// C API. If you are writing C++ code, use the namespaced versions.

typedef enum legion_error_t {
  NO_ERROR = 0,
  ERROR_RESERVED_REDOP_ID = 1,
  ERROR_DUPLICATE_REDOP_ID = 2,
  ERROR_RESERVED_TYPE_HANDLE = 3,
  ERROR_DUPLICATE_TYPE_HANDLE = 4,
  ERROR_DUPLICATE_FIELD_ID = 5,
  ERROR_PARENT_TYPE_HANDLE_NONEXISTENT = 6,
  ERROR_MISSING_PARENT_FIELD_ID = 7,
  ERROR_RESERVED_PROJECTION_ID = 8,
  ERROR_DUPLICATE_PROJECTION_ID = 9,
  ERROR_UNREGISTERED_VARIANT = 10,
  ERROR_USE_REDUCTION_REGION_REQ = 11,
  ERROR_INVALID_ACCESSOR_REQUESTED = 12,
  ERROR_PHYSICAL_REGION_UNMAPPED = 13,
  ERROR_RESERVED_TASK_ID = 14,
  ERROR_INVALID_ARG_MAP_DESTRUCTION = 15,
  ERROR_RESERVED_MAPPING_ID = 16,
  ERROR_BAD_INDEX_PRIVILEGES = 17,
  ERROR_BAD_FIELD_PRIVILEGES = 18,
  ERROR_BAD_REGION_PRIVILEGES = 19,
  ERROR_BAD_PARTITION_PRIVILEGES = 20,
  ERROR_BAD_PARENT_INDEX = 21,
  ERROR_BAD_INDEX_PATH = 22,
  ERROR_BAD_PARENT_REGION = 23,
  ERROR_BAD_REGION_PATH = 24,
  ERROR_BAD_PARTITION_PATH = 25,
  ERROR_BAD_FIELD = 26,
  ERROR_BAD_REGION_TYPE = 27,
  ERROR_INVALID_TYPE_HANDLE = 28,
  ERROR_LEAF_TASK_VIOLATION = 29,
  ERROR_INVALID_REDOP_ID = 30,
  ERROR_REDUCTION_INITIAL_VALUE_MISMATCH = 31,
  ERROR_INVALID_UNMAP_OP = 32,
  ERROR_INVALID_DUPLICATE_MAPPING = 33,
  ERROR_INVALID_REGION_ARGUMENT_INDEX = 34,
  ERROR_INVALID_MAPPING_ACCESS = 35,
  ERROR_STALE_INLINE_MAPPING_ACCESS = 36,
  ERROR_INVALID_INDEX_SPACE_PARENT = 37,
  ERROR_INVALID_INDEX_PART_PARENT = 38,
  ERROR_INVALID_INDEX_SPACE_COLOR = 39,
  ERROR_INVALID_INDEX_PART_COLOR = 40,
  ERROR_INVALID_INDEX_SPACE_HANDLE = 41,
  ERROR_INVALID_INDEX_PART_HANDLE = 42,
  ERROR_FIELD_SPACE_FIELD_MISMATCH = 43,
  ERROR_INVALID_INSTANCE_FIELD = 44,
  ERROR_DUPLICATE_INSTANCE_FIELD = 45,
  ERROR_TYPE_INST_MISMATCH = 46,
  ERROR_TYPE_INST_MISSIZE = 47,
  ERROR_INVALID_INDEX_SPACE_ENTRY = 48,
  ERROR_INVALID_INDEX_PART_ENTRY = 49,
  ERROR_INVALID_FIELD_SPACE_ENTRY = 50,
  ERROR_INVALID_REGION_ENTRY = 51,
  ERROR_INVALID_PARTITION_ENTRY = 52,
  ERROR_ALIASED_INTRA_TASK_REGIONS = 53,
  ERROR_MAX_FIELD_OVERFLOW = 54,
  ERROR_MISSING_TASK_COLLECTION = 55,
  ERROR_INVALID_IDENTITY_PROJECTION_USE = 56,
  ERROR_INVALID_PROJECTION_ID = 57,
  ERROR_NON_DISJOINT_PARTITION = 58,
  ERROR_BAD_PROJECTION_USE = 59,
  ERROR_INDEPENDENT_SLICES_VIOLATION = 60,
  ERROR_INVALID_REGION_HANDLE = 61,
  ERROR_INVALID_PARTITION_HANDLE = 62,
  ERROR_VIRTUAL_MAP_IN_LEAF_TASK = 63,
  ERROR_LEAF_MISMATCH = 64,
  ERROR_INVALID_PROCESSOR_SELECTION = 65,
  ERROR_INVALID_VARIANT_SELECTION = 66,
  ERROR_INVALID_MAPPER_OUTPUT = 67,
  ERROR_UNINITIALIZED_REDUCTION = 68,
  ERROR_INVALID_INDEX_DOMAIN = 69,
  ERROR_INVALID_INDEX_PART_DOMAIN = 70,
  ERROR_DISJOINTNESS_TEST_FAILURE = 71,
  ERROR_NON_DISJOINT_TASK_REGIONS = 72,
  ERROR_INVALID_FIELD_ACCESSOR_PRIVILEGES = 73,
  ERROR_INVALID_PREMAPPED_REGION_LOCATION = 74,
  ERROR_IDEMPOTENT_MISMATCH = 75,
  ERROR_INVALID_MAPPER_ID = 76,
  ERROR_INVALID_TREE_ENTRY = 77,
  ERROR_SEPARATE_UTILITY_PROCS = 78,
  ERROR_MAXIMUM_NODES_EXCEEDED = 79,
  ERROR_MAXIMUM_PROCS_EXCEEDED = 80,
  ERROR_INVALID_TASK_ID = 81,
  ERROR_INVALID_MAPPER_DOMAIN_SLICE = 82,
  ERROR_UNFOLDABLE_REDUCTION_OP = 83,
  ERROR_INVALID_INLINE_ID = 84,
  ERROR_ILLEGAL_MUST_PARALLEL_INLINE = 85,
  ERROR_RETURN_SIZE_MISMATCH = 86,
  ERROR_ACCESSING_EMPTY_FUTURE = 87,
  ERROR_ILLEGAL_PREDICATE_FUTURE = 88,
  ERROR_COPY_REQUIREMENTS_MISMATCH = 89,
  ERROR_INVALID_COPY_FIELDS_SIZE = 90,
  ERROR_COPY_SPACE_MISMATCH = 91,
  ERROR_INVALID_COPY_PRIVILEGE = 92,
  ERROR_INVALID_PARTITION_COLOR = 93,
  ERROR_EXCEEDED_MAX_CONTEXTS = 94,
  ERROR_ACQUIRE_MISMATCH = 95,
  ERROR_RELEASE_MISMATCH = 96,
  ERROR_INNER_LEAF_MISMATCH = 97,
  ERROR_INVALID_FIELD_PRIVILEGES = 98,
  ERROR_ILLEGAL_NESTED_TRACE = 99,
  ERROR_UNMATCHED_END_TRACE = 100,
  ERROR_CONFLICTING_PARENT_MAPPING_DEADLOCK = 101,
  ERROR_CONFLICTING_SIBLING_MAPPING_DEADLOCK = 102,
  ERROR_INVALID_PARENT_REQUEST = 103,
  ERROR_INVALID_FIELD_ID = 104,
  ERROR_NESTED_MUST_EPOCH = 105,
  ERROR_UNMATCHED_MUST_EPOCH = 106,
  ERROR_MUST_EPOCH_FAILURE = 107,
  ERROR_DOMAIN_DIM_MISMATCH = 108,
  ERROR_INVALID_PROCESSOR_NAME = 109,
  ERROR_INVALID_INDEX_SUBSPACE_REQUEST = 110,
  ERROR_INVALID_INDEX_SUBPARTITION_REQUEST = 111,
  ERROR_INVALID_FIELD_SPACE_REQUEST = 112,
  ERROR_INVALID_LOGICAL_SUBREGION_REQUEST = 113,
  ERROR_INVALID_LOGICAL_SUBPARTITION_REQUEST = 114,
  ERROR_ALIASED_REGION_REQUIREMENTS = 115,
  ERROR_MISSING_DEFAULT_PREDICATE_RESULT = 116,
  ERROR_PREDICATE_RESULT_SIZE_MISMATCH = 117,
  ERROR_MPI_INTEROPERABILITY_NOT_CONFIGURED = 118,
  ERROR_TRACING_ALLOCATION_WITH_SEPARATE = 119,
  ERROR_EMPTY_INDEX_PARTITION = 120,
  ERROR_INCONSISTENT_SEMANTIC_TAG = 121,
  ERROR_INVALID_SEMANTIC_TAG = 122,
  ERROR_DUMMY_CONTEXT_OPERATION = 123,
  ERROR_INVALID_CONTEXT_CONFIGURATION = 124,
  ERROR_INDEX_TREE_MISMATCH = 125,
  ERROR_INDEX_PARTITION_ANCESTOR = 126,
  ERROR_INVALID_PENDING_CHILD = 127,
  ERROR_ILLEGAL_FILE_ATTACH = 128,
  ERROR_ILLEGAL_ALLOCATOR_REQUEST = 129,
  ERROR_ILLEGAL_DETACH_OPERATION = 130,
  ERROR_NO_PROCESSORS = 131,
  ERROR_ILLEGAL_REDUCTION_VIRTUAL_MAPPING = 132,
  ERROR_INVALID_MAPPED_REGION_LOCATION = 133,
  ERROR_RESERVED_SERDEZ_ID = 134,
  ERROR_DUPLICATE_SERDEZ_ID = 135,
  ERROR_INVALID_SERDEZ_ID = 136,
  ERROR_TRACE_VIOLATION = 137,
  ERROR_INVALID_TARGET_PROC = 138,
  ERROR_INCOMPLETE_TRACE = 139,
  ERROR_STATIC_CALL_POST_RUNTIME_START = 140,
  ERROR_ILLEGAL_GLOBAL_VARIANT_REGISTRATION = 141,
  ERROR_ILLEGAL_USE_OF_NON_GLOBAL_VARIANT = 142,
  ERROR_RESERVED_CONSTRAINT_ID = 143,
  ERROR_INVALID_CONSTRAINT_ID = 144,
  ERROR_DUPLICATE_CONSTRAINT_ID = 145,
  ERROR_ILLEGAL_WAIT_FOR_SHUTDOWN = 146,
  ERROR_DEPRECATED_METHOD_USE = 147,
  ERROR_MAX_APPLICATION_TASK_ID_EXCEEDED = 148,
  ERROR_MAX_APPLICATION_MAPPER_ID_EXCEEDED = 149,
  ERROR_INVALID_ARGUMENTS_TO_MAPPER_RUNTIME = 150,
  ERROR_INVALID_MAPPER_SYNCHRONIZATION = 151,
  ERROR_ILLEGAL_PARTIAL_ACQUISITION = 152,
  ERROR_ILLEGAL_INTERFERING_RESTRICTIONS = 153,
  ERROR_ILLEGAL_PARTIAL_RESTRICTION = 154,
  ERROR_ILLEGAL_INTERFERING_ACQUISITIONS = 155,
  ERROR_UNRESTRICTED_ACQUIRE = 156,
  ERROR_UNACQUIRED_RELEASE = 157,
  ERROR_UNATTACHED_DETACH = 158,
  ERROR_INVALID_PROJECTION_RESULT = 159,
  ERROR_ILLEGAL_IMPLICIT_MAPPING = 160,
  ERROR_INNER_TASK_VIOLATION = 161,
  ERROR_REQUEST_FOR_EMPTY_FUTURE = 162,
  ERROR_ILLEGAL_REMAP_IN_STATIC_TRACE = 163,
  ERROR_DYNAMIC_TYPE_MISMATCH = 164,
  ERROR_MISSING_LOCAL_VARIABLE = 165,
  ERROR_ACCESSOR_PRIVILEGE_CHECK = 166,
  ERROR_ACCESSOR_BOUNDS_CHECK = 167,
  ERROR_DUPLICATE_MPI_CONFIG = 168,
  ERROR_UNKNOWN_MAPPABLE = 169,
  ERROR_DEPRECATED_PROJECTION = 170,
  ERROR_ILLEGAL_PARTIAL_ACQUIRE = 171,
  ERROR_ILLEGAL_INTERFERING_RESTRICTON = 172,
  ERROR_ILLEGAL_INTERFERING_ACQUIRE = 173,
  ERROR_ILLEGAL_REDUCTION_REQUEST = 175,
  ERROR_PROJECTION_REGION_REQUIREMENTS = 180,
  ERROR_REQUIREMENTS_INVALID_REGION = 181,
  ERROR_FIELD_NOT_VALID_FIELD = 182,
  ERROR_INSTANCE_FIELD_PRIVILEGE = 183,
  ERROR_ILLEGAL_REQUEST_VIRTUAL_INSTANCE = 185,
  ERROR_PARENT_TASK_INLINE = 186,
  ERROR_REGION_NOT_SUBREGION = 189,
  ERROR_REGION_REQUIREMENT_INLINE = 190,
  ERROR_PRIVILEGES_FOR_REGION = 191,
  ERROR_MISSING_INSTANCE_FIELD = 195,
  ERROR_NUMBER_SOURCE_REQUIREMENTS = 204,
  ERROR_COPY_SOURCE_REQUIREMENTS = 205,
  ERROR_COPY_DESTINATION_REQUIREMENT = 206,
  ERROR_COPY_LAUNCHER_INDEX = 208,
  ERROR_DESTINATION_INDEX_SPACE = 209,
  ERROR_ALIASED_REQION_REQUIREMENTS = 210,
  ERROR_REQUEST_INVALID_REGION = 212,
  ERROR_INSTANCE_FIELD_DUPLICATE = 215,
  ERROR_PARENT_TASK_COPY = 216,
  ERROR_REGION_REQUIREMENT_COPY = 220,
  ERROR_SOURCE_REGION_REQUIREMENT = 232,
  ERROR_DESTINATION_REGION_REQUIREMENT = 233,
  ERROR_COPY_SOURCE_REQUIREMENT = 235,
  ERROR_INDEX_SPACE_COPY = 237,
  ERROR_DESTINATION_INDEX_SPACE2 = 240,
  ERROR_MAPPER_FAILED_ACQUIRE = 245,
  ERROR_FIELD_NOT_VALID = 248,
  ERROR_PARENT_TASK_ACQUIRE = 249,
  ERROR_REGION_REQUIREMENT_ACQUIRE = 253,
  ERROR_PARENT_TASK_RELEASE = 257,
  ERROR_REGION_REQUIREMENT_RELEASE = 261,
  ERROR_MUST_EPOCH_DEPENDENCE = 267,
  ERROR_PARENT_TASK_PARTITION = 268,
  ERROR_PARENT_TASK_FILL = 273,
  ERROR_REGION_REQUIREMENT_FILL = 278,
  ERROR_PRIVILEGES_REGION_SUBSET = 279,
  ERROR_INDEX_SPACE_FILL = 281,
  ERROR_ILLEGAL_FILE_ATTACHMENT = 284,
  ERROR_REGION_REQUIREMENT_ATTACH = 293,
  ERROR_PARENT_TASK_DETACH = 295,
  ERROR_PREDICATED_TASK_LAUNCH = 296,
  ERROR_MAPPER_REQUESTED_EXECUTION = 297,
  ERROR_PARENT_TASK_TASK = 298,
  ERROR_INDEX_SPACE_NOTSUBSPACE = 299,
  ERROR_PRIVILEGES_INDEX_SPACE = 300,
  ERROR_PROJECTION_REGION_REQUIREMENT = 303,
  ERROR_NONDISJOINT_PARTITION_SELECTED = 304,
  ERROR_PARTITION_NOT_SUBPARTITION = 312,
  ERROR_REGION_REQUIREMENT_TASK = 313,
  ERROR_PRIVILEGES_REGION_NOTSUBSET = 314,
  ERROR_PRIVILEGES_PARTITION_NOTSUBSET = 315,
  ERROR_INVALID_LOCATION_CONSTRAINT = 344,
  ERROR_ALIASED_INTERFERING_REGION = 356,
  ERROR_REDUCTION_OPERATION_INDEX = 357,
  ERROR_PREDICATED_INDEX_TASK = 358,
  ERROR_INDEX_SPACE_TASK = 359,
  ERROR_TRACE_VIOLATION_RECORDED = 363,
  ERROR_TRACE_VIOLATION_OPERATION = 364,
  ERROR_INVALID_MAPPER_REQUEST = 366,
  ERROR_ILLEGAL_RUNTIME_REMAPPING = 377,
  ERROR_UNABLE_FIND_TASK_LOCAL = 378,
  ERROR_INDEXPARTITION_NOT_SAME_INDEX_TREE = 379,
  ERROR_TASK_ATTEMPTED_ALLOCATE_FILED = 386,
  ERROR_EXCEEDED_MAXIMUM_NUMBER_LOCAL_FIELDS = 387,
  ERROR_UNABLE_ALLOCATE_LOCAL_FIELD = 388,
  ERROR_TASK_ATTEMPTED_ALLOCATE_FIELD = 389,
  ERROR_PREDICATED_TASK_LAUNCH_FOR_TASK = 392,
  ERROR_PREDICATED_INDEX_TASK_LAUNCH = 393,
  ERROR_ATTEMPTED_INLINE_MAPPING_REGION = 395,
  ERROR_ATTEMPTED_ATTACH_HDF5 = 397,
  ERROR_ILLEGAL_PREDICATE_CREATION = 399,
  ERROR_ILLEGAL_END_TRACE_CALL = 402,
  ERROR_UNMATCHED_END_STATIC_TRACE = 403,
  ERROR_ILLEGAL_END_STATIC_TRACE = 404,
  ERROR_ILLEGAL_ACQUIRE_OPERATION = 405,
  ERROR_ILLEGAL_RELEASE_OPERATION = 406,
  ERROR_TASK_FAILED_END_TRACE = 408,
  ERROR_ILLEGAL_INDEX_SPACE_CREATION = 410,
  ERROR_UMATCHED_END_TRACE = 411,
  ERROR_ILLEGAL_NESTED_STATIC_TRACE = 412,
  ERROR_ILLEGAL_UNION_INDEX_SPACES = 414,
  ERROR_ILLEGAL_INTERSECT_INDEX_SPACES = 415,
  ERROR_ILLEGAL_SUBTRACT_INDEX_SPACES = 416,
  ERROR_ILLEGAL_INDEX_SPACE_DELETION = 417,
  ERROR_ILLEGAL_INDEX_PARTITION_DELETION = 418,
  ERROR_ILLEGAL_EQUAL_PARTITION_CREATION = 419,
  ERROR_ILLEGAL_UNION_PARTITION_CREATION = 420,
  ERROR_ILLEGAL_INTERSECTION_PARTITION_CREATION = 421,
  ERROR_ILLEGAL_DIFFERENCE_PARTITION_CREATION = 422,
  ERROR_ILLEGAL_CREATE_CROSS_PRODUCT_PARTITION = 423,
  ERROR_ILLEGAL_CREATE_ASSOCIATION = 424,
  ERROR_ILLEGAL_CREATE_RESTRICTED_PARTITION = 425,
  ERROR_ILLEGAL_PARTITION_FIELD = 426,
  ERROR_ILLEGAL_PARTITION_IMAGE = 427,
  ERROR_ILLEGAL_PARTITION_IMAGE_RANGE = 428,
  ERROR_ILLEGAL_PARTITION_PREIMAGE = 429,
  ERROR_ILLEGAL_PARTITION_PREIMAGE_RANGE = 430,
  ERROR_ILLEGAL_CREATE_PENDING_PARTITION = 431,
  ERROR_ILLEGAL_CREATE_INDEX_SPACE_UNION = 432,
  ERROR_ILLEGAL_CREATE_INDEX_SPACE_INTERSECTION = 434,
  ERROR_ILLEGAL_CREATE_INDEX_SPACE_DIFFERENCE = 436,
  ERROR_ILLEGAL_CREATE_FIELD_SPACE = 437,
  ERROR_ILLEGAL_DESTROY_FIELD_SPACE = 438,
  ERROR_ILLEGAL_NONLOCAL_FIELD_ALLOCATION = 439,
  ERROR_ILLEGAL_FIELD_DESTRUCTION = 440,
  ERROR_ILLEGAL_NONLOCAL_FIELD_ALLOCATION2 = 441,
  ERROR_ILLEGAL_FIELD_DESTRUCTION2 = 442,
  ERROR_ILLEGAL_REGION_CREATION = 443,
  ERROR_ILLEGAL_REGION_DESTRUCTION = 444,
  ERROR_ILLEGAL_PARTITION_DESTRUCTION = 445,
  ERROR_ILLEGAL_CREATE_FIELD_ALLOCATION = 447,
  ERROR_ILLEGAL_EXECUTE_TASK_CALL = 448,
  ERROR_ILLEGAL_EXECUTE_INDEX_SPACE = 449,
  ERROR_ILLEGAL_MAP_REGION = 451,
  ERROR_ILLEGAL_REMAP_OPERATION = 452,
  ERROR_ILLEGAL_UNMAP_OPERATION = 453,
  ERROR_ILLEGAL_FILL_OPERATION_CALL = 454,
  ERROR_ILLEGAL_INDEX_FILL_OPERATION_CALL = 455,
  ERROR_ILLEGAL_COPY_FILL_OPERATION_CALL = 456,
  ERROR_ILLEGAL_INDEX_COPY_OPERATION = 457,
  ERROR_ILLEGAL_ATTACH_RESOURCE_OPERATION = 460,
  ERROR_ILLEGAL_DETACH_RESOURCE_OPERATION = 461,
  ERROR_ILLEGAL_LEGION_EXECUTE_MUST_EPOCH = 462,
  ERROR_ILLEGAL_TIMING_MEASUREMENT = 463,
  ERROR_ILLEGAL_LEGION_MAPPING_FENCE_CALL = 464,
  ERROR_ILLEGAL_LEGION_EXECUTION_FENCE_CALL = 465,
  ERROR_ILLEGAL_LEGION_COMPLETE_FRAME_CALL = 466,
  ERROR_ILLEGAL_GET_PREDICATE_FUTURE = 469,
  ERROR_ILLEGAL_LEGION_BEGIN_TRACE = 470,
  ERROR_ILLEGAL_LEGION_END_TRACE = 471,
  ERROR_ILLEGAL_LEGION_BEGIN_STATIC_TRACE = 472,
  ERROR_ILLEGAL_LEGION_END_STATIC_TRACE = 473,
  ERROR_INVALID_PHYSICAL_TRACING = 474,
  ERROR_INCOMPLETE_PHYSICAL_TRACING = 475,
  ERROR_PHYSICAL_TRACING_UNSUPPORTED_OP = 476,
  ERROR_PHYSICAL_TRACING_REMOTE_MAPPING = 477,
  ERROR_PARENT_INDEX_PARTITION_REQUESTED = 478,
  ERROR_FIELD_SPACE_HAS_NO_FIELD = 479,
  ERROR_PARENT_LOGICAL_PARTITION_REQUESTED = 480,
  ERROR_INVALID_REQUEST_FOR_INDEXSPACE = 481,
  ERROR_UNABLE_FIND_ENTRY = 482,
  ERROR_INVALID_REQUEST_INDEXPARTITION = 484,
  ERROR_INVALID_REQUEST_FIELDSPACE = 487,
  ERROR_INVALID_REQUEST_LOGICALREGION = 490,
  ERROR_INVALID_REQUEST_LOGICALPARTITION = 492,
  ERROR_INVALID_REQUEST_TREE_ID = 493,
  ERROR_UNABLE_FIND_TOPLEVEL_TREE = 495,
  ERROR_ILLEGAL_DUPLICATE_REQUEST_ALLOCATOR = 502,
  ERROR_ILLEGAL_DUPLICATE_FIELD_ID = 510,
  ERROR_EXCEEDED_MAXIMUM_NUMBER_ALLOCATED_FIELDS = 511,
  ERROR_ILLEGAL_NOT_PREDICATE_CREATION = 533,
  ERROR_PARENT_TASK_ATTACH = 534,
  ERROR_INVALID_REGION_ARGUMENTS = 535,
  ERROR_INVALID_MAPPER_CONTENT = 536,
  ERROR_INVALID_DUPLICATE_MAPPER = 537,
  ERROR_INVALID_UNLOCK_MAPPER = 538,
  ERROR_UNKNOWN_PROFILER_OPTION = 539,
  ERROR_MISSING_PROFILER_OPTION = 540,
  ERROR_INVALID_PROFILER_SERIALIZER = 541,
  ERROR_INVALID_PROFILER_FILE = 542, 
  ERROR_ILLEGAL_LAYOUT_CONSTRAINT = 543,
  ERROR_UNSUPPORTED_LAYOUT_CONSTRAINT = 544,
  ERROR_ACCESSOR_FIELD_SIZE_CHECK = 545,
  ERROR_ATTACH_OPERATION_MISSING_POINTER = 546,
  ERROR_RESERVED_VARIANT_ID = 547,
  ERROR_NON_DENSE_RECTANGLE = 548,
  ERROR_LIBRARY_COUNT_MISMATCH = 549, 
  ERROR_MPI_INTEROP_MISCONFIGURATION = 550,
  ERROR_NUMBER_SRC_INDIRECT_REQUIREMENTS = 551,
  ERROR_NUMBER_DST_INDIRECT_REQUIREMENTS = 552,
  ERROR_COPY_GATHER_REQUIREMENT = 553,
  ERROR_COPY_SCATTER_REQUIREMENT = 554,
  ERROR_MAPPER_SYNCHRONIZATION = 555,
  ERROR_DUPLICATE_VARIANT_REGISTRATION = 556,
  ERROR_ILLEGAL_IMPLICIT_TOP_LEVEL_TASK = 557,
  

  LEGION_WARNING_FUTURE_NONLEAF = 1000,
  LEGION_WARNING_BLOCKING_EMPTY = 1001,
  LEGION_WARNING_WAITING_ALL_FUTURES = 1002,
  LEGION_WARNING_WAITING_REGION = 1003,
  LEGION_WARNING_MISSING_REGION_WAIT = 1004,
  LEGION_WARNING_NONLEAF_ACCESSOR = 1005,
  LEGION_WARNING_UNMAPPED_ACCESSOR = 1006,
  LEGION_WARNING_READ_DISCARD = 1007,
  LEGION_WARNING_MISSING_PROC_CONSTRAINT = 1008,
  LEGION_WARNING_DYNAMIC_PROJECTION_REG = 1009,
  LEGION_WARNING_DUPLICATE_MPI_CONFIG = 1010,
  LEGION_WARNING_NEW_PROJECTION_FUNCTORS = 1011,
  LEGION_WARNING_IGNORE_MEMORY_REQUEST = 1012,
  LEGION_WARNING_NOT_COPY = 1013,
  LEGION_WARNING_REGION_REQUIREMENT_INLINE = 1014,
  LEGION_WARNING_MAPPER_FAILED_ACQUIRE = 1015,
  LEGION_WARNING_SOURCE_REGION_REQUIREMENT = 1016,
  LEGION_WARNING_DESTINATION_REGION_REQUIREMENT = 1017,
  LEGION_WARNING_REGION_REQUIREMENTS_INDEX = 1019,
  LEGION_WARNING_PRIVILEGE_FIELDS_ACQUIRE = 1020,
  LEGION_WARNING_PRIVILEGE_FIELDS_RELEASE = 1021,
  LEGION_WARNING_FILE_ATTACH_OPERATION = 1022,
  LEGION_WARNING_HDF5_ATTACH_OPERATION = 1023,
  LEGION_WARNING_REGION_REQUIREMENT_TASK = 1024,
  LEGION_WARNING_EMPTY_OUTPUT_TARGET = 1026,
  LEGION_WARNING_IGNORING_SPURIOUS_TARGET = 1027,
  LEGION_WARNING_IGNORING_PROCESSOR_REQUEST = 1028,
  LEGION_WARNING_MAPPER_REQUESTED_COMPOSITE = 1030,
  LEGION_WARNING_MAPPER_REQUESTED_INLINE = 1031,
  LEGION_WARNING_REGION_REQUIREMENTS_INDIVIDUAL = 1032,
  LEGION_WARNING_IGNORING_ACQUIRE_REQUEST = 1035,
  LEGION_WARNING_WAITING_FUTURE_NONLEAF = 1047,
  LEGION_WARNING_RUNTIME_UNMAPPING_REMAPPING = 1050,
  LEGION_WARNING_IGNORING_EMPTY_INDEX_TASK_LAUNCH = 1058,
  LEGION_WARNING_REGION_REQUIREMENT_OPERATION_USING = 1071,
  LEGION_WARNING_MAPPER_REQUESTED_PROFILING = 1082,
  LEGION_WARNING_REGION_TREE_STATE_LOGGING = 1083,
  LEGION_WARNING_DISJOINTNESS_VERIFICATION = 1084,
  LEGION_WARNING_IGNORING_EMPTY_INDEX_SPACE_FILL = 1085,
  LEGION_WARNING_IGNORING_EMPTY_INDEX_SPACE_COPY = 1086,
  LEGION_WARNING_VARIANT_TASK_NOT_MARKED = 1087,
  LEGION_WARNING_MAPPER_REQUESTED_POST = 1088,
  LEGION_WARNING_IGNORING_RELEASE_REQUEST = 1089,
  LEGION_WARNING_PRUNE_DEPTH_EXCEEDED = 1090,
  LEGION_WARNING_GENERIC_ACCESSOR = 1091, 
  LEGION_WARNING_UNUSED_PROFILING_FILE_NAME = 1092,
  LEGION_WARNING_INVALID_PRIORITY_CHANGE = 1093,
  LEGION_WARNING_EXTERNAL_ATTACH_OPERATION = 1094,
  LEGION_WARNING_EXTERNAL_GARBAGE_PRIORITY = 1095,
  LEGION_WARNING_MAPPER_INVALID_INSTANCE = 1096,
  LEGION_WARNING_NON_REPLAYABLE_COUNT_EXCEEDED = 1097,
  
  
  LEGION_FATAL_MUST_EPOCH_NOADDRESS = 2000,
  LEGION_FATAL_MUST_EPOCH_NOTASKS = 2001,
  LEGION_FATAL_DEFAULT_MAPPER_ERROR = 2002,
  LEGION_FATAL_SHIM_MAPPER_SUPPORT = 2006,
  LEGION_FATAL_UNKNOWN_FIELD_ID = 2007,
  LEGION_FATAL_RESTRICTED_SIMULTANEOUS = 2008,
  LEGION_FATAL_EXCEEDED_LIBRARY_ID_OFFSET = 2009,
  
  
}  legion_error_t;

// enum and namepsaces don't really get along well
// We would like to make these associations explicit
// but the python cffi parser is stupid as hell
typedef enum legion_privilege_mode_t {
  NO_ACCESS       = 0x00000000, 
  READ_PRIV       = 0x00000001,
  READ_ONLY       = 0x00000001, // READ_PRIV,
  WRITE_PRIV      = 0x00000002,
  REDUCE_PRIV     = 0x00000004,
  REDUCE          = 0x00000004, // REDUCE_PRIV,
  READ_WRITE      = 0x00000007, // READ_PRIV | WRITE_PRIV | REDUCE_PRIV,
  DISCARD_MASK    = 0x10000000, // For marking we don't need inputs
  WRITE_ONLY      = 0x10000002, // WRITE_PRIV | DISCARD_MASK,
  WRITE_DISCARD   = 0x10000007, // READ_WRITE | DISCARD_MASK,
} legion_privilege_mode_t;

typedef enum legion_allocate_mode_t {
  NO_MEMORY       = 0x00000000,
  ALLOCABLE       = 0x00000001,
  FREEABLE        = 0x00000002,
  MUTABLE         = 0x00000003,
  REGION_CREATION = 0x00000004,
  REGION_DELETION = 0x00000008,
  ALL_MEMORY      = 0x0000000F,
} legion_allocate_mode_t;

typedef enum legion_coherence_property_t {
  EXCLUSIVE    = 0,
  ATOMIC       = 1,
  SIMULTANEOUS = 2,
  RELAXED      = 3,
} legion_coherence_property_t;

// Optional region requirement flags
typedef enum legion_region_flags_t {
  NO_FLAG             = 0x00000000,
  VERIFIED_FLAG       = 0x00000001,
  NO_ACCESS_FLAG      = 0x00000002, // Deprecated, user SpecializedConstraint
  RESTRICTED_FLAG     = 0x00000004,
  MUST_PREMAP_FLAG    = 0x00000008,
  // For non-trivial projection functions: 
  // tell the runtime the write is complete,
  // will be ignored for non-index space launches
  // and for privileges that aren't WRITE
  // Note that if you use this incorrectly it could
  // break the correctness of your code so be sure
  // you know what you are doing
  COMPLETE_PROJECTION_WRITE_FLAG = 0x00000010,
} legion_region_flags_t;

typedef enum legion_projection_type_t {
  SINGULAR, // a single logical region
  PART_PROJECTION, // projection from a partition
  REG_PROJECTION, // projection from a region
} legion_projection_type_t;
// For backwards compatibility
typedef legion_projection_type_t legion_handle_type_t;

typedef enum legion_partition_kind_t {
  DISJOINT_KIND, // disjoint and unknown
  ALIASED_KIND, // aliased and unknown
  COMPUTE_KIND, // unknown and unknown
  DISJOINT_COMPLETE_KIND, // disjoint and complete
  ALIASED_COMPLETE_KIND, // aliased and complete
  COMPUTE_COMPLETE_KIND, // unknown and complete
  DISJOINT_INCOMPLETE_KIND, // disjoint and incomplete
  ALIASED_INCOMPLETE_KIND, // aliased and incomplete
  COMPUTE_INCOMPLETE_KIND, // unknown and incomplete
} legion_partition_kind_t;

typedef enum legion_external_resource_t {
  EXTERNAL_POSIX_FILE,
  EXTERNAL_HDF5_FILE,
  EXTERNAL_INSTANCE,
} legion_external_resource_t;

typedef enum legion_timing_measurement_t {
  MEASURE_SECONDS,
  MEASURE_MICRO_SECONDS,
  MEASURE_NANO_SECONDS,
} legion_timing_measurement_t;

typedef enum legion_dependence_type_t {
  NO_DEPENDENCE = 0,
  TRUE_DEPENDENCE = 1,
  ANTI_DEPENDENCE = 2, // WAR or WAW with Write-Only privilege
  ATOMIC_DEPENDENCE = 3,
  SIMULTANEOUS_DEPENDENCE = 4,
} legion_dependence_type_t;

enum {
  NAME_SEMANTIC_TAG = 0,
  FIRST_AVAILABLE_SEMANTIC_TAG = 1,
};

typedef enum legion_execution_constraint_t {
  ISA_CONSTRAINT = 0, // instruction set architecture
  PROCESSOR_CONSTRAINT = 1, // processor kind constraint
  RESOURCE_CONSTRAINT = 2, // physical resources
  LAUNCH_CONSTRAINT = 3, // launch configuration
  COLOCATION_CONSTRAINT = 4, // region requirements in same instance
} legion_execution_constraint_t;

typedef enum legion_layout_constraint_t {
  SPECIALIZED_CONSTRAINT = 0, // normal or speicalized (e.g. reduction-fold)
  MEMORY_CONSTRAINT = 1, // constraint on the kind of memory
  FIELD_CONSTRAINT = 2, // ordering of fields
  ORDERING_CONSTRAINT = 3, // ordering of dimensions
  SPLITTING_CONSTRAINT = 4, // splitting of dimensions 
  DIMENSION_CONSTRAINT = 5, // dimension size constraint
  ALIGNMENT_CONSTRAINT = 6, // alignment of a field
  OFFSET_CONSTRAINT = 7, // offset of a field
  POINTER_CONSTRAINT = 8, // pointer of a field
} legion_layout_constraint_t;

typedef enum legion_equality_kind_t {
  LT_EK = 0, // <
  LE_EK = 1, // <=
  GT_EK = 2, // >
  GE_EK = 3, // >=
  EQ_EK = 4, // ==
  NE_EK = 5, // !=
} legion_equality_kind_t;

typedef enum legion_dimension_kind_t {
  DIM_X = 0, // first logical index space dimension
  DIM_Y = 1, // second logical index space dimension
  DIM_Z = 2, // ...
  // field dimension (this is here for legacy reasons: Regent has hard coded it)
  DIM_W = 3, // fourth logical index space dimension
  DIM_V = 4, // fifth logical index space dimension
  DIM_U = 5, // ...
  DIM_T = 6,
  DIM_S = 7,
  DIM_R = 8,
  DIM_F = 9, 
  INNER_DIM_X = 10, // inner dimension for tiling X
  OUTER_DIM_X = 11, // outer dimension for tiling X
  INNER_DIM_Y = 12, // ...
  OUTER_DIM_Y = 13,
  INNER_DIM_Z = 14,
  OUTER_DIM_Z = 15,
  INNER_DIM_W = 16,
  OUTER_DIM_W = 17,
  INNER_DIM_V = 18,
  OUTER_DIM_V = 19,
  INNER_DIM_U = 20,
  OUTER_DIM_U = 21,
  INNER_DIM_T = 22,
  OUTER_DIM_T = 23,
  INNER_DIM_S = 24,
  OUTER_DIM_S = 25,
  INNER_DIM_R = 26,
  OUTER_DIM_R = 27,
} legion_dimension_kind_t;

// Make all flags 1-hot encoding so we can logically-or them together
typedef enum legion_isa_kind_t {
  // Top-level ISA Kinds
  X86_ISA   = 0x00000001,
  ARM_ISA   = 0x00000002,
  PPC_ISA   = 0x00000004, // Power PC
  PTX_ISA   = 0x00000008, // auto-launch by runtime
  CUDA_ISA  = 0x00000010, // run on CPU thread bound to CUDA context
  LUA_ISA   = 0x00000020, // run on Lua processor
  TERRA_ISA = 0x00000040, // JIT to target processor kind
  LLVM_ISA  = 0x00000080, // JIT to target processor kind
  GL_ISA    = 0x00000100, // run on CPU thread with OpenGL context
  // x86 Vector Instructions
  SSE_ISA   = 0x00000200,
  SSE2_ISA  = 0x00000400,
  SSE3_ISA  = 0x00000800,
  SSE4_ISA  = 0x00001000,
  AVX_ISA   = 0x00002000,
  AVX2_ISA  = 0x00004000,
  FMA_ISA   = 0x00008000,
  // PowerPC Vector Insructions
  VSX_ISA   = 0x00010000,
  // GPU variants
  SM_10_ISA = 0x00020000,
  SM_20_ISA = 0x00040000,
  SM_30_ISA = 0x00080000,
  SM_35_ISA = 0x00100000,
  // ARM Vector Instructions
  NEON_ISA  = 0x00200000,
} legion_isa_kind_t;

typedef enum legion_resource_constraint_t {
  L1_CACHE_SIZE = 0,
  L2_CACHE_SIZE = 1,
  L3_CACHE_SIZE = 2,
  L1_CACHE_ASSOCIATIVITY = 3,
  L2_CACHE_ASSOCIATIVITY = 4,
  L3_CACHE_ASSOCIATIVITY = 5,
  REGISTER_FILE_SIZE = 6,
  SHARED_MEMORY_SIZE = 7,
  TEXTURE_CACHE_SIZE = 8,
  CONSTANT_CACHE_SIZE = 9,
  NAMED_BARRIERS = 10,
  SM_COUNT = 11, // total SMs on the device
  MAX_OCCUPANCY = 12, // max warps per SM
} legion_resource_constraint_t;

typedef enum legion_launch_constraint_t {
  CTA_SHAPE = 0,
  GRID_SHAPE = 1,
  DYNAMIC_SHARED_MEMORY = 2,
  REGISTERS_PER_THREAD = 3,
  CTAS_PER_SM = 4,
  NAMED_BARRIERS_PER_CTA = 5,
} legion_launch_constraint_t;

typedef enum legion_specialized_constraint_t {
  NO_SPECIALIZE = 0,
  NORMAL_SPECIALIZE = 1,
  REDUCTION_FOLD_SPECIALIZE = 2,
  REDUCTION_LIST_SPECIALIZE = 3,
  VIRTUAL_SPECIALIZE = 4,
  // All file types must go below here, everything else above
  GENERIC_FILE_SPECIALIZE = 5,
  HDF5_FILE_SPECIALIZE = 6,
} legion_specialized_constraint_t;

// Keep this in sync with Domain::MAX_RECT_DIM in legion_domain.h
#define LEGION_MAX_POINT_DIM (LEGION_MAX_DIM)
#define LEGION_MAX_RECT_DIM  (LEGION_MAX_DIM)
typedef enum legion_domain_max_rect_dim_t {
  MAX_POINT_DIM = LEGION_MAX_POINT_DIM,
  MAX_RECT_DIM = LEGION_MAX_RECT_DIM,
} legion_domain_max_rect_dim_t;

//==========================================================================
//                                Types
//==========================================================================

typedef realm_processor_kind_t legion_processor_kind_t;
typedef realm_memory_kind_t legion_memory_kind_t;
typedef realm_reduction_op_id_t legion_reduction_op_id_t;
typedef realm_custom_serdez_id_t legion_custom_serdez_id_t;
typedef realm_address_space_t legion_address_space_t;
typedef realm_file_mode_t legion_file_mode_t;
typedef realm_id_t legion_proc_id_t;
typedef realm_id_t legion_memory_id_t;
typedef int legion_task_priority_t;
typedef int legion_garbage_collection_priority_t;
typedef long long legion_coord_t;
typedef unsigned int legion_color_t;
typedef unsigned int legion_field_id_t;
typedef unsigned int legion_trace_id_t;
typedef unsigned int legion_mapper_id_t;
typedef unsigned int legion_context_id_t;
typedef unsigned int legion_instance_id_t;
typedef unsigned int legion_type_tag_t;
typedef unsigned int legion_index_space_id_t;
typedef unsigned int legion_index_partition_id_t;
typedef unsigned int legion_index_tree_id_t;
typedef unsigned int legion_field_space_id_t;
typedef unsigned int legion_generation_id_t;
typedef unsigned int legion_type_handle;
typedef unsigned int legion_projection_id_t;
typedef unsigned int legion_sharding_id_t;
typedef unsigned int legion_region_tree_id_t;
typedef unsigned int legion_tunable_id_t;
typedef unsigned int legion_local_variable_id_t;
typedef unsigned int legion_replication_id_t;
typedef unsigned int legion_shard_id_t;
typedef unsigned int legion_variant_id_t;
typedef unsigned long long legion_distributed_id_t;
typedef unsigned long legion_mapping_tag_id_t;
typedef unsigned long legion_code_descriptor_id_t;
typedef unsigned long legion_semantic_tag_t;
typedef unsigned long long legion_unique_id_t;
typedef unsigned long long legion_version_id_t;
typedef unsigned long long legion_projection_epoch_id_t;
typedef realm_task_func_id_t legion_task_id_t;
typedef unsigned long legion_layout_constraint_id_t;
typedef long long legion_internal_color_t;

#endif // __LEGION_CONFIG_H__

