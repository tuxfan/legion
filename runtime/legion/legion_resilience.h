//sri
/* Copyright 2017 Stanford University, NVIDIA Corporation
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

#ifndef __LEGION_RESILIENCE_H__
#define __LEGION_RESILIENCE_H__

#include "realm.h"
#include "utilities.h"
#include "legion_types.h"
#include "legion_utilities.h"
//#include "realm/resilience.h"
#include "lowlevel_config.h"

#include <cassert>
#include <deque>
#include <algorithm>
#include <sstream>

namespace Legion {
  namespace Internal { 

    // XXX: Make sure these typedefs are consistent with Realm
    typedef ::legion_lowlevel_barrier_timestamp_t timestamp_t;
    typedef Realm::Processor::Kind ProcKind;
    typedef Realm::Memory::Kind MemKind;
    typedef ::legion_lowlevel_id_t ProcID;
    typedef ::legion_lowlevel_id_t MemID;
    typedef ::legion_lowlevel_id_t InstID;

//    class LegionResilienceMarker {
//    public:
//      LegionResilienceMarker(const char* name);
//      ~LegionResilienceMarker();
//      void mark_stop();
//    private:
//      const char* name;
//      bool stopped;
//      Processor proc;
//      timestamp_t start, stop;
//    };
//
//    class LegionResilienceDesc {
//    public:
//      struct MessageDesc {
//      public:
//        unsigned kind;
//        const char *name;
//      };
//      struct MapperCallDesc {
//      public:
//        unsigned kind;
//        const char *name;
//      };
//      struct RuntimeCallDesc {
//      public:
//        unsigned kind;
//        const char *name;
//      };
//      struct MetaDesc {
//      public:
//        unsigned kind;
//        const char *name;
//      };
//      struct OpDesc {
//      public:
//        unsigned kind;
//        const char *name;
//      };
//      struct ProcDesc {
//      public:
//        ProcID proc_id;
//        ProcKind kind;
//      };
//      struct MemDesc {
//      public:
//        MemID mem_id;
//        MemKind kind;
//        unsigned long long capacity;
//      };
//    };
//
//    class LegionResilienceInstance {
//    public:
//      struct TaskKind {
//      public:
//        TaskID task_id;
//        const char *name;
//        bool overwrite;
//      };
//      struct TaskVariant {
//      public:
//        TaskID task_id;
//        VariantID variant_id;
//        const char *name;
//      };
//      struct OperationInstance {
//      public:
//        UniqueID op_id;
//        unsigned kind;
//      };
//      struct MultiTask {
//      public:
//        UniqueID op_id;
//        TaskID task_id;
//      };
//      struct SliceOwner {
//      public:
//        UniqueID parent_id;
//        UniqueID op_id;
//      };
//      struct WaitInfo {
//      public:
//        timestamp_t wait_start, wait_ready, wait_end;
//      };
//      struct TaskInfo {
//      public:
//        UniqueID op_id;
//        VariantID variant_id;
//        ProcID proc_id;
//        timestamp_t create, ready, start, stop;
//        std::deque<WaitInfo> wait_intervals;
//      };
//      struct MetaInfo {
//      public:
//        UniqueID op_id;
//        unsigned lg_id;
//        ProcID proc_id;
//        timestamp_t create, ready, start, stop;
//        std::deque<WaitInfo> wait_intervals;
//      };
//      struct CopyInfo {
//      public:
//        UniqueID op_id;
//        MemID src, dst;
//        unsigned long long size;
//        timestamp_t create, ready, start, stop;
//      };
//      struct FillInfo {
//      public:
//        UniqueID op_id;
//        MemID dst;
//        timestamp_t create, ready, start, stop;
//      };
//      struct InstCreateInfo {
//      public:
//        UniqueID op_id;
//        InstID inst_id;
//        timestamp_t create; // time of HLR creation request
//      };
//      struct InstUsageInfo {
//      public:
//        UniqueID op_id;
//        InstID inst_id;
//        MemID mem_id;
//        unsigned long long size;
//      };
//      struct InstTimelineInfo {
//      public:
//        UniqueID op_id;
//        InstID inst_id;
//        timestamp_t create, destroy;
//      };
//      struct MessageInfo {
//      public:
//        MessageKind kind;
//        timestamp_t start, stop;
//        ProcID proc_id;
//      };
//      struct MapperCallInfo {
//      public:
//        MappingCallKind kind;
//        UniqueID op_id;
//        timestamp_t start, stop;
//        ProcID proc_id;
//      };
//      struct RuntimeCallInfo {
//      public:
//        RuntimeCallKind kind;
//        timestamp_t start, stop;
//        ProcID proc_id;
//      };
//    public:
//      LegionResilienceInstance(LegionResilienceiler *owner);
//      LegionResilienceInstance(const LegionResilienceInstance &rhs);
//      ~LegionResilienceInstance(void);
//    public:
//      LegionResilienceInstance& operator=(const LegionResilienceInstance &rhs);
//    public:
//      void register_task_kind(TaskID task_id, const char *name, bool overwrite);
//      void register_task_variant(TaskID task_id,
//                                 VariantID variant_id, 
//                                 const char *variant_name);
//      void register_operation(Operation *op);
//      void register_multi_task(Operation *op, TaskID kind);
//      void register_slice_owner(UniqueID pid, UniqueID id);
//    public:
//      void process_task(VariantID variant_id, UniqueID op_id, 
//                  Realm::ResiliencyMeasurements::OperationTimeline *timeline,
//                  Realm::ResiliencyMeasurements::OperationProcessorUsage *usage,
//                  Realm::ResiliencyMeasurements::OperationEventWaits *waits);
//      void process_meta(size_t id, UniqueID op_id,
//                  Realm::ResiliencyMeasurements::OperationTimeline *timeline,
//                  Realm::ResiliencyMeasurements::OperationProcessorUsage *usage,
//                  Realm::ResiliencyMeasurements::OperationEventWaits *waits);
//      void process_message(
//                  Realm::ResiliencyMeasurements::OperationTimeline *timeline,
//                  Realm::ResiliencyMeasurements::OperationProcessorUsage *usage,
//                  Realm::ResiliencyMeasurements::OperationEventWaits *waits);
//      void process_copy(UniqueID op_id,
//                  Realm::ResiliencyMeasurements::OperationTimeline *timeline,
//                  Realm::ResiliencyMeasurements::OperationMemoryUsage *usage);
//      void process_fill(UniqueID op_id,
//                  Realm::ResiliencyMeasurements::OperationTimeline *timeline,
//                  Realm::ResiliencyMeasurements::OperationMemoryUsage *usage);
//      void process_inst_create(UniqueID op_id, PhysicalInstance inst,
//                               timestamp_t create);
//      void process_inst_usage(UniqueID op_id,
//                  Realm::ResiliencyMeasurements::InstanceMemoryUsage *usage);
//      void process_inst_timeline(UniqueID op_id,
//                  Realm::ResiliencyMeasurements::InstanceTimeline *timeline);
//    public:
//      void record_message(Processor proc, MessageKind kind, timestamp_t start,
//                          timestamp_t stop);
//      void record_mapper_call(Processor proc, MappingCallKind kind, 
//                              UniqueID uid, timestamp_t start,
//                              timestamp_t stop);
//      void record_runtime_call(Processor proc, RuntimeCallKind kind,
//                               timestamp_t start, timestamp_t stop);
//    public:
//      void dump_state(LegionResilienceSerializer *serializer);
//    private:
//      LegionResilient *const owner;
//      std::deque<TaskKind>          task_kinds;
//      std::deque<TaskVariant>       task_variants;
//      std::deque<OperationInstance> operation_instances;
//      std::deque<MultiTask>         multi_tasks;
//      std::deque<SliceOwner>        slice_owners;
//    private:
//      std::deque<TaskInfo> task_infos;
//      std::deque<MetaInfo> meta_infos;
//      std::deque<CopyInfo> copy_infos;
//      std::deque<FillInfo> fill_infos;
//      std::deque<InstCreateInfo> inst_create_infos;
//      std::deque<InstUsageInfo> inst_usage_infos;
//      std::deque<InstTimelineInfo> inst_timeline_infos;
//    private:
//      std::deque<MessageInfo> message_infos;
//      std::deque<MapperCallInfo> mapper_call_infos;
//      std::deque<RuntimeCallInfo> runtime_call_infos;
//    };
//
    class LegionResilient {
//    public:
//      enum ResiliencyKind {
//        LEGION_RESILIENCE_TASK,
//        LEGION_RESILIENCE_META,
//        LEGION_RESILIENCE_MESSAGE,
//        LEGION_RESILIENCE_COPY,
//        LEGION_RESILIENCE_FILL,
//        LEGION_RESILIENCE_INST,
//      };
//      struct ResiliencyInfo {
//      public:
//        ResiliencyInfo(ResiliencyKind k)
//          : kind(k) { }
//      public:
//        ResiliencyKind kind;
//        size_t id;
//        UniqueID op_id;
//      };
    public:
      // Statically known information passed through the constructor
      // so that it can be deduplicated
      LegionResilient();
      LegionResilient(const LegionResilient &rhs);
      ~LegionResilient(void);
    public:
      LegionResilient& operator=(const LegionResilient &rhs);
//    public:
//      // Dynamically created things must be registered at runtime
//      // Tasks
//      void register_task_kind(TaskID task_id, const char *task_name, 
//                              bool overwrite);
//      void register_task_variant(TaskID task_id,
//                                 VariantID var_id,
//                                 const char *variant_name);
//      // Operations
//      void register_operation(Operation *op);
//      void register_multi_task(Operation *op, TaskID task_id);
//      void register_slice_owner(UniqueID pid, UniqueID id);
    public:
      void add_task_request();//Realm::ResiliencyRequestSet &requests, 
                            //TaskID tid, SingleTask *task);
//      void add_meta_request(Realm::ResiliencyRequestSet &requests,
//                            LgTaskID tid, Operation *op);
//      void add_copy_request(Realm::ResiliencyRequestSet &requests, 
//                            Operation *op);
//      void add_fill_request(Realm::ResiliencyRequestSet &requests,
//                            Operation *op);
//      void add_inst_request(Realm::ResiliencyRequestSet &requests,
//                            Operation *op);
//      // Adding a message resiliency request is a static method
//      // because we might not have a resilient on the local node
//      static void add_message_request(Realm::ResiliencyRequestSet &requests,
//                                      Processor remote_target);
//    public:
//      // Alternate versions of the one above with op ids
//      void add_task_request(Realm::ResiliencyRequestSet &requests, 
//                            TaskID tid, UniqueID uid);
//      void add_meta_request(Realm::ResiliencyRequestSet &requests,
//                            LgTaskID tid, UniqueID uid);
//      void add_copy_request(Realm::ResiliencyRequestSet &requests, 
//                            UniqueID uid);
//      void add_fill_request(Realm::ResiliencyRequestSet &requests,
//                            UniqueID uid);
//      void add_inst_request(Realm::ResiliencyRequestSet &requests,
//                            UniqueID uid);
//    public:
//      // Process low-level runtime resiliency results
      void process_results(Processor p, const void *buffer, size_t size);
    public:
      // Dump all the results
      void finalize(void);
//    public:
//      void record_instance_creation(PhysicalInstance inst, Memory memory,
//                                    UniqueID op_id, timestamp_t create);
//    public:
//      void record_message_kinds(const char *const *const message_names,
//                                unsigned int num_message_kinds);
//      void record_message(MessageKind kind, timestamp_t start,
//                          timestamp_t stop);
//    public:
//      void record_mapper_call_kinds(const char *const *const mapper_call_names,
//                                    unsigned int num_mapper_call_kinds);
//      void record_mapper_call(MappingCallKind kind, UniqueID uid,
//                              timestamp_t start, timestamp_t stop);
//    public:
//      void record_runtime_call_kinds(const char *const *const runtime_calls,
//                                     unsigned int num_runtime_call_kinds);
//      void record_runtime_call(RuntimeCallKind kind, timestamp_t start,
//                               timestamp_t stop);
//    public:
//      const Processor target_proc;
//      inline bool has_outstanding_requests(void)
//        { return total_outstanding_requests != 0; }
//    public:
//      inline void increment_total_outstanding_requests(void)
//        { __sync_fetch_and_add(&total_outstanding_requests,1); }
//      inline void decrement_total_outstanding_requests(void)
//        { __sync_fetch_and_sub(&total_outstanding_requests,1); }
//    private:
//      void create_thread_local_resiliency_instance(void);
//    private:
//      LegionResilienceSerializer* serializer;
//      Reservation resilient_lock;
//      std::vector<LegionResilienceInstance*> instances;
//      unsigned total_outstanding_requests;
    };

  }; // namespace Internal
}; // namespace Legion

#endif // __LEGION_RESILIENCE_H__

