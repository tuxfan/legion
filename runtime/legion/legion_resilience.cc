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

#include "legion.h"
#include "cmdline.h"
#include "legion_ops.h"
#include "legion_tasks.h"
#include "legion_resilience.h"

#include <cstring>
#include <cstdlib>

namespace Legion {
  namespace Internal {
    //--------------------------------------------------------------------------
    LegionResilient::LegionResilient()
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    LegionResilient::LegionResilient(const LegionResilient &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    LegionResilient::~LegionResilient(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    LegionResilient& LegionResilient::operator=(const LegionResilient &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    

    //--------------------------------------------------------------------------
    void LegionResilient::add_task_request() //Realm::ResiliencyRequestSet &requests,
                                          //TaskID tid, SingleTask *task)
    //--------------------------------------------------------------------------
    {
//      increment_total_outstanding_requests();
//      ResiliencyInfo info(LEGION_PROF_TASK); 
//      info.id = tid;
//      info.op_id = task->get_unique_id();
//      Realm::ResiliencyRequest &req = requests.add_request((target_proc.exists())
//                        ? target_proc : Processor::get_executing_processor(),
//                        LG_LEGION_RESILIENCY_ID, &info, sizeof(info));
//      req.add_measurement<
//                Realm::ResiliencyMeasurements::OperationTimeline>();
    }

    ////--------------------------------------------------------------------------
    //void LegionResilient::add_meta_request(Realm::ResiliencyRequestSet &requests,
    //                                      LgTaskID tid, Operation *op)
    ////--------------------------------------------------------------------------
    //{
    //  increment_total_outstanding_requests();
    //  ResiliencyInfo info(LEGION_PROF_META); 
    //  info.id = tid;
    //  info.op_id = (op != NULL) ? op->get_unique_op_id() : 0;
    //  Realm::ResiliencyRequest &req = requests.add_request((target_proc.exists())
    //                    ? target_proc : Processor::get_executing_processor(),
    //                    LG_LEGION_RESILIENCY_ID, &info, sizeof(info));
    //  req.add_measurement<
    //            Realm::ResiliencyMeasurements::OperationTimeline>();
    //  req.add_measurement<
    //            Realm::ResiliencyMeasurements::OperationProcessorUsage>();
    //  req.add_measurement<
    //            Realm::ResiliencyMeasurements::OperationEventWaits>();
    //}

    ////--------------------------------------------------------------------------
    ///*static*/ void LegionResilient::add_message_request(
    //              Realm::ResiliencyRequestSet &requests, Processor remote_target)
    ////--------------------------------------------------------------------------
    //{
    //  // Don't increment here, we'll increment on the remote side since we
    //  // that is where we know the resilient is going to handle the results
    //  ResiliencyInfo info(LEGION_PROF_MESSAGE);
    //  Realm::ResiliencyRequest &req = requests.add_request(remote_target,
    //                    LG_LEGION_RESILIENCY_ID, &info, sizeof(info));
    //  req.add_measurement<
    //            Realm::ResiliencyMeasurements::OperationTimeline>();
    //  req.add_measurement<
    //            Realm::ResiliencyMeasurements::OperationProcessorUsage>();
    //  req.add_measurement<
    //            Realm::ResiliencyMeasurements::OperationEventWaits>();
    //}

    ////--------------------------------------------------------------------------
    //void LegionResilient::add_copy_request(Realm::ResiliencyRequestSet &requests,
    //                                      Operation *op)
    ////--------------------------------------------------------------------------
    //{
    //  ResiliencyInfo info(LEGION_PROF_COPY); 
    //  // No ID here
    //  info.op_id = (op != NULL) ? op->get_unique_op_id() : 0;
    //  Realm::ResiliencyRequest &req = requests.add_request((target_proc.exists())
    //                    ? target_proc : Processor::get_executing_processor(),
    //                    LG_LEGION_RESILIENCY_ID, &info, sizeof(info));
    //  req.add_measurement<
    //            Realm::ResiliencyMeasurements::OperationTimeline>();
    //  req.add_measurement<
    //            Realm::ResiliencyMeasurements::OperationMemoryUsage>();
    //}

    ////--------------------------------------------------------------------------
    //void LegionResilient::add_fill_request(Realm::ResiliencyRequestSet &requests,
    //                                      Operation *op)
    ////--------------------------------------------------------------------------
    //{
    //  // wonchan: don't track fill operations for the moment
    //  // as their requests and responses do not exactly match
    //  //increment_total_outstanding_requests();
    //  ResiliencyInfo info(LEGION_PROF_FILL);
    //  // No ID here
    //  info.op_id = (op != NULL) ? op->get_unique_op_id() : 0;
    //  Realm::ResiliencyRequest &req = requests.add_request((target_proc.exists())
    //                    ? target_proc : Processor::get_executing_processor(),
    //                    LG_LEGION_RESILIENCY_ID, &info, sizeof(info));
    //  req.add_measurement<
    //            Realm::ResiliencyMeasurements::OperationTimeline>();
    //  req.add_measurement<
    //            Realm::ResiliencyMeasurements::OperationMemoryUsage>();
    //}

    ////--------------------------------------------------------------------------
    //void LegionResilient::add_inst_request(Realm::ResiliencyRequestSet &requests,
    //                                      Operation *op)
    ////--------------------------------------------------------------------------
    //{
    //  ResiliencyInfo info(LEGION_PROF_INST); 
    //  // No ID here
    //  info.op_id = (op != NULL) ? op->get_unique_op_id() : 0;
    //  Realm::ResiliencyRequest &req = requests.add_request((target_proc.exists())
    //                    ? target_proc : Processor::get_executing_processor(),
    //                    LG_LEGION_RESILIENCY_ID, &info, sizeof(info));
    //  req.add_measurement<
    //            Realm::ResiliencyMeasurements::InstanceTimeline>();
    //  req.add_measurement<
    //            Realm::ResiliencyMeasurements::InstanceMemoryUsage>();
    //}

    //////--------------------------------------------------------------------------
    ////void LegionResilient::add_task_request(Realm::ResiliencyRequestSet &requests,
    ////                                      TaskID tid, UniqueID uid)
    //////--------------------------------------------------------------------------
    ////{
    ////  increment_total_outstanding_requests();
    ////  ResiliencyInfo info(LEGION_PROF_TASK); 
    ////  info.id = tid;
    ////  info.op_id = uid;
    ////  Realm::ResiliencyRequest &req = requests.add_request((target_proc.exists())
    ////                    ? target_proc : Processor::get_executing_processor(),
    ////                    LG_LEGION_RESILIENCY_ID, &info, sizeof(info));
    ////  req.add_measurement<
    ////            Realm::ResiliencyMeasurements::OperationTimeline>();
    ////  req.add_measurement<
    ////            Realm::ResiliencyMeasurements::OperationProcessorUsage>();
    ////  req.add_measurement<
    ////            Realm::ResiliencyMeasurements::OperationEventWaits>();
    ////}

    ////--------------------------------------------------------------------------
    //void LegionResilient::add_meta_request(Realm::ResiliencyRequestSet &requests,
    //                                      LgTaskID tid, UniqueID uid)
    ////--------------------------------------------------------------------------
    //{
    //  increment_total_outstanding_requests();
    //  ResiliencyInfo info(LEGION_PROF_META); 
    //  info.id = tid;
    //  info.op_id = uid;
    //  Realm::ResiliencyRequest &req = requests.add_request((target_proc.exists())
    //                    ? target_proc : Processor::get_executing_processor(),
    //                    LG_LEGION_RESILIENCY_ID, &info, sizeof(info));
    //  req.add_measurement<
    //            Realm::ResiliencyMeasurements::OperationTimeline>();
    //  req.add_measurement<
    //            Realm::ResiliencyMeasurements::OperationProcessorUsage>();
    //  req.add_measurement<
    //            Realm::ResiliencyMeasurements::OperationEventWaits>();
    //}

    ////--------------------------------------------------------------------------
    //void LegionResilient::add_copy_request(Realm::ResiliencyRequestSet &requests,
    //                                      UniqueID uid)
    ////--------------------------------------------------------------------------
    //{
    //  ResiliencyInfo info(LEGION_PROF_COPY); 
    //  // No ID here
    //  info.op_id = uid;
    //  Realm::ResiliencyRequest &req = requests.add_request((target_proc.exists())
    //                    ? target_proc : Processor::get_executing_processor(),
    //                    LG_LEGION_RESILIENCY_ID, &info, sizeof(info));
    //  req.add_measurement<
    //            Realm::ResiliencyMeasurements::OperationTimeline>();
    //  req.add_measurement<
    //            Realm::ResiliencyMeasurements::OperationMemoryUsage>();
    //}

    ////--------------------------------------------------------------------------
    //void LegionResilient::add_fill_request(Realm::ResiliencyRequestSet &requests,
    //                                      UniqueID uid)
    ////--------------------------------------------------------------------------
    //{
    //  // wonchan: don't track fill operations for the moment
    //  // as their requests and responses do not exactly match
    //  //increment_total_outstanding_requests();
    //  ResiliencyInfo info(LEGION_PROF_FILL);
    //  // No ID here
    //  info.op_id = uid;
    //  Realm::ResiliencyRequest &req = requests.add_request((target_proc.exists())
    //                    ? target_proc : Processor::get_executing_processor(),
    //                    LG_LEGION_RESILIENCY_ID, &info, sizeof(info));
    //  req.add_measurement<
    //            Realm::ResiliencyMeasurements::OperationTimeline>();
    //  req.add_measurement<
    //            Realm::ResiliencyMeasurements::OperationMemoryUsage>();
    //}

    ////--------------------------------------------------------------------------
    //void LegionResilient::add_inst_request(Realm::ResiliencyRequestSet &requests,
    //                                      UniqueID uid)
    ////--------------------------------------------------------------------------
    //{
    //  ResiliencyInfo info(LEGION_PROF_INST); 
    //  // No ID here
    //  info.op_id = uid;
    //  // Instances use two resiliency requests so that we can get MemoryUsage
    //  // right away - the Timeline doesn't come until we delete the instance
    //  Processor p = (target_proc.exists() 
    //                    ? target_proc : Processor::get_executing_processor());
    //  Realm::ResiliencyRequest &req1 = requests.add_request(p,
    //                    LG_LEGION_RESILIENCY_ID, &info, sizeof(info));
    //  req1.add_measurement<
    //             Realm::ResiliencyMeasurements::InstanceMemoryUsage>();
    //  Realm::ResiliencyRequest &req2 = requests.add_request(p,
    //                    LG_LEGION_RESILIENCY_ID, &info, sizeof(info));
    //  req2.add_measurement<
    //             Realm::ResiliencyMeasurements::InstanceTimeline>();
    //}

    //--------------------------------------------------------------------------
    void LegionResilient::process_results(Processor p, const void *buffer,
                                         size_t size)
    //--------------------------------------------------------------------------
    {
        volatile static int i=10;
				++i;
//#ifdef LEGION_PROF_SELF_PROFILE
//      long long t_start = Realm::Clock::current_time_in_nanoseconds();
//#endif
//      if (thread_local_resiliency_instance == NULL)
//        create_thread_local_resiliency_instance();
//      Realm::ResiliencyResponse response(buffer, size);
//#ifdef DEBUG_LEGION
//      assert(response.user_data_size() == sizeof(ResiliencyInfo));
//#endif
//      const ResiliencyInfo *info = (const ResiliencyInfo*)response.user_data();
//      switch (info->kind)
//      {
//        case LEGION_PROF_TASK:
//          {
//#ifdef DEBUG_LEGION
//            assert(response.has_measurement<
//                Realm::ResiliencyMeasurements::OperationTimeline>());
//            //assert(response.has_measurement<
//            //    Realm::ResiliencyMeasurements::OperationProcessorUsage>());
//#endif
//            Realm::ResiliencyMeasurements::OperationTimeline *timeline = 
//              response.get_measurement<
//                    Realm::ResiliencyMeasurements::OperationTimeline>();
////            Realm::ResiliencyMeasurements::OperationProcessorUsage *usage = 
////              response.get_measurement<
////                    Realm::ResiliencyMeasurements::OperationProcessorUsage>();
////            Realm::ResiliencyMeasurements::OperationEventWaits *waits = 
////              response.get_measurement<
////                    Realm::ResiliencyMeasurements::OperationEventWaits>();
//            // Ignore anything that was predicated false for now
//            if (true) //usage != NULL)
//              thread_local_resiliency_instance->process_task(info->id, 
//                  info->op_id, timeline, usage, waits);
//            if (timeline != NULL)
//              delete timeline;
//            if (timeline != NULL)
//              delete usage;
//            decrement_total_outstanding_requests();
//            break;
//          }
////        case LEGION_PROF_META:
////          {
////#ifdef DEBUG_LEGION
////            assert(response.has_measurement<
////                Realm::ResiliencyMeasurements::OperationTimeline>());
////            assert(response.has_measurement<
////                Realm::ResiliencyMeasurements::OperationProcessorUsage>());
////#endif
////            Realm::ResiliencyMeasurements::OperationTimeline *timeline = 
////              response.get_measurement<
////                    Realm::ResiliencyMeasurements::OperationTimeline>();
////            Realm::ResiliencyMeasurements::OperationProcessorUsage *usage = 
////              response.get_measurement<
////                    Realm::ResiliencyMeasurements::OperationProcessorUsage>();
////            Realm::ResiliencyMeasurements::OperationEventWaits *waits = 
////              response.get_measurement<
////                    Realm::ResiliencyMeasurements::OperationEventWaits>();
////            // Ignore anything that was predicated false for now
////            if (usage != NULL)
////              thread_local_resiliency_instance->process_meta(info->id, 
////                  info->op_id, timeline, usage, waits);
////            if (timeline != NULL)
////              delete timeline;
////            if (usage != NULL)
////              delete usage;
////            decrement_total_outstanding_requests();
////            break;
////          }
////        case LEGION_PROF_MESSAGE:
////          {
////#ifdef DEBUG_LEGION
////            assert(response.has_measurement<
////                Realm::ResiliencyMeasurements::OperationTimeline>());
////            assert(response.has_measurement<
////                Realm::ResiliencyMeasurements::OperationProcessorUsage>());
////#endif
////            Realm::ResiliencyMeasurements::OperationTimeline *timeline = 
////              response.get_measurement<
////                    Realm::ResiliencyMeasurements::OperationTimeline>();
////            Realm::ResiliencyMeasurements::OperationProcessorUsage *usage = 
////              response.get_measurement<
////                    Realm::ResiliencyMeasurements::OperationProcessorUsage>();
////            Realm::ResiliencyMeasurements::OperationEventWaits *waits = 
////              response.get_measurement<
////                    Realm::ResiliencyMeasurements::OperationEventWaits>();
////            if (usage != NULL)
////              thread_local_resiliency_instance->process_message(timeline, 
////                  usage, waits);
////            if (timeline != NULL)
////              delete timeline;
////            if (usage != NULL)
////              delete usage;
////            decrement_total_outstanding_requests();
////            break;
////          }
////        case LEGION_PROF_COPY:
////          {
////#ifdef DEBUG_LEGION
////            assert(response.has_measurement<
////                Realm::ResiliencyMeasurements::OperationTimeline>());
////            assert(response.has_measurement<
////                Realm::ResiliencyMeasurements::OperationMemoryUsage>());
////#endif
////            Realm::ResiliencyMeasurements::OperationTimeline *timeline = 
////              response.get_measurement<
////                    Realm::ResiliencyMeasurements::OperationTimeline>();
////            Realm::ResiliencyMeasurements::OperationMemoryUsage *usage = 
////              response.get_measurement<
////                    Realm::ResiliencyMeasurements::OperationMemoryUsage>();
////            // Ignore anything that was predicated false for now
////            if (usage != NULL)
////              thread_local_resiliency_instance->process_copy(info->op_id,
////                                                            timeline, usage);
////            if (timeline != NULL)
////              delete timeline;
////            if (usage != NULL)
////              delete usage;
////            break;
////          }
////        case LEGION_PROF_FILL:
////          {
////#ifdef DEBUG_LEGION
////            assert(response.has_measurement<
////                Realm::ResiliencyMeasurements::OperationTimeline>());
////            assert(response.has_measurement<
////                Realm::ResiliencyMeasurements::OperationMemoryUsage>());
////#endif
////            Realm::ResiliencyMeasurements::OperationTimeline *timeline = 
////              response.get_measurement<
////                    Realm::ResiliencyMeasurements::OperationTimeline>();
////            Realm::ResiliencyMeasurements::OperationMemoryUsage *usage = 
////              response.get_measurement<
////                    Realm::ResiliencyMeasurements::OperationMemoryUsage>();
////            // Ignore anything that was predicated false for now
////            if (usage != NULL)
////              thread_local_resiliency_instance->process_fill(info->op_id,
////                                                            timeline, usage);
////            if (timeline != NULL)
////              delete timeline;
////            if (usage != NULL)
////              delete usage;
////            // wonchan: don't track fill operations for the moment
////            // as their requests and responses do not exactly match
////            //decrement_total_outstanding_requests();
////            break;
////          }
////        case LEGION_PROF_INST:
////          {
////	    // Record data based on which measurements we got back this time
////	    if (response.has_measurement<
////                Realm::ResiliencyMeasurements::InstanceTimeline>())
////	    {
////	      Realm::ResiliencyMeasurements::InstanceTimeline *timeline = 
////                response.get_measurement<
////                      Realm::ResiliencyMeasurements::InstanceTimeline>();
////	      thread_local_resiliency_instance->process_inst_timeline(
////								info->op_id,
////								timeline);
////	      delete timeline;
////	    }
////	    if (response.has_measurement<
////                Realm::ResiliencyMeasurements::InstanceMemoryUsage>())
////	    {
////	      Realm::ResiliencyMeasurements::InstanceMemoryUsage *usage = 
////                response.get_measurement<
////                      Realm::ResiliencyMeasurements::InstanceMemoryUsage>();
////	      thread_local_resiliency_instance->process_inst_usage(info->op_id,
////								  usage);
////	      delete usage;
////	    }
////            break;
////          }
//        default:
//          assert(false);
//      }
//#ifdef LEGION_PROF_SELF_PROFILE
//      long long t_stop = Realm::Clock::current_time_in_nanoseconds();
//      thread_local_resiliency_instance->record_resiliencetask(p, info->op_id, 
//                                                       t_start, t_stop);
//#endif
    }

    //--------------------------------------------------------------------------
    void LegionResilient::finalize(void)
    //--------------------------------------------------------------------------
    {
//      for (std::vector<LegionResilienceInstance*>::const_iterator it = 
//            instances.begin(); it != instances.end(); it++) {
//        (*it)->dump_state(serializer);
//      }  
    }

//    //--------------------------------------------------------------------------
//    void LegionResilient::record_instance_creation(PhysicalInstance inst,
//                       Memory memory, UniqueID op_id, unsigned long long create)
//    //--------------------------------------------------------------------------
//    {
//      if (thread_local_resiliency_instance == NULL)
//        create_thread_local_resiliency_instance();
//      thread_local_resiliency_instance->process_inst_create(op_id, inst, create);
//    }
//
//    //--------------------------------------------------------------------------
//    void LegionResilient::record_message_kinds(const char *const *const
//                                  message_names, unsigned int num_message_kinds)
//    //--------------------------------------------------------------------------
//    {
//      for (unsigned idx = 0; idx < num_message_kinds; idx++)
//      {
//        LegionResilienceDesc::MessageDesc message_desc;
//        message_desc.kind = idx;
//        message_desc.name = message_names[idx];
//        serializer->serialize(message_desc);
//      }
//    }
//
//    //--------------------------------------------------------------------------
//    void LegionResilient::record_message(MessageKind kind, 
//                                        unsigned long long start,
//                                        unsigned long long stop)
//    //--------------------------------------------------------------------------
//    {
//      Processor current = Processor::get_executing_processor();
//      if (thread_local_resiliency_instance == NULL)
//        create_thread_local_resiliency_instance();
//      thread_local_resiliency_instance->record_message(current, kind, 
//                                                      start, stop);
//    }
//
//    //--------------------------------------------------------------------------
//    void LegionResilient::record_mapper_call_kinds(const char *const *const
//                               mapper_call_names, unsigned int num_mapper_calls)
//    //--------------------------------------------------------------------------
//    {
//      for (unsigned idx = 0; idx < num_mapper_calls; idx++)
//      {
//        LegionResilienceDesc::MapperCallDesc mapper_call_desc;
//        mapper_call_desc.kind = idx;
//        mapper_call_desc.name = mapper_call_names[idx];
//        serializer->serialize(mapper_call_desc);
//      }
//    }
//
//    //--------------------------------------------------------------------------
//    void LegionResilient::record_mapper_call(MappingCallKind kind, UniqueID uid,
//                              unsigned long long start, unsigned long long stop)
//    //--------------------------------------------------------------------------
//    {
//      Processor current = Processor::get_executing_processor();
//      if (thread_local_resiliency_instance == NULL)
//        create_thread_local_resiliency_instance();
//      thread_local_resiliency_instance->record_mapper_call(current, kind, uid, 
//                                                   start, stop);
//    }
//
//    //--------------------------------------------------------------------------
//    void LegionResilient::record_runtime_call_kinds(const char *const *const
//                             runtime_call_names, unsigned int num_runtime_calls)
//    //--------------------------------------------------------------------------
//    {
//      for (unsigned idx = 0; idx < num_runtime_calls; idx++)
//      {
//        LegionResilienceDesc::RuntimeCallDesc runtime_call_desc;
//        runtime_call_desc.kind = idx;
//        runtime_call_desc.name = runtime_call_names[idx];
//        serializer->serialize(runtime_call_desc);
//      }
//    }
//
//    //--------------------------------------------------------------------------
//    void LegionResilient::record_runtime_call(RuntimeCallKind kind,
//                              unsigned long long start, unsigned long long stop)
//    //--------------------------------------------------------------------------
//    {
//      Processor current = Processor::get_executing_processor();
//      if (thread_local_resiliency_instance == NULL)
//        create_thread_local_resiliency_instance();
//      thread_local_resiliency_instance->record_runtime_call(current, kind, 
//                                                           start, stop);
//    }
//
//    //--------------------------------------------------------------------------
//    void LegionResilient::create_thread_local_resiliency_instance(void)
//    //--------------------------------------------------------------------------
//    {
//      thread_local_resiliency_instance = new LegionResilienceInstance(this);
//      // Task the lock and save the reference
//      AutoLock p_lock(resilient_lock);
//      instances.push_back(thread_local_resiliency_instance);
//    }
//
//    //--------------------------------------------------------------------------
//    DetailedResilient::DetailedResilient(Runtime *runtime, RuntimeCallKind call)
//      : resilient(runtime->resilient), call_kind(call), start_time(0)
//    //--------------------------------------------------------------------------
//    {
//      if (resilient != NULL)
//        start_time = Realm::Clock::current_time_in_nanoseconds();
//    }
//
//    //--------------------------------------------------------------------------
//    DetailedResilient::DetailedResilient(const DetailedResilient &rhs)
//      : resilient(rhs.resilient), call_kind(rhs.call_kind)
//    //--------------------------------------------------------------------------
//    {
//      // should never be called
//      assert(false);
//    }
//
//    //--------------------------------------------------------------------------
//    DetailedResilient::~DetailedResilient(void)
//    //--------------------------------------------------------------------------
//    {
//      if (resilient != NULL)
//      {
//        unsigned long long stop_time = 
//          Realm::Clock::current_time_in_nanoseconds();
//        resilient->record_runtime_call(call_kind, start_time, stop_time);
//      }
//    }
//
//    //--------------------------------------------------------------------------
//    DetailedResilient& DetailedResilient::operator=(const DetailedResilient &rhs)
//    //--------------------------------------------------------------------------
//    {
//      // should never be called
//      assert(false);
//      return *this;
//    }

  }; // namespace Internal
}; // namespace Legion

