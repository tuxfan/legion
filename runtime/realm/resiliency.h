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

// profiling infrastructure for Realm tasks, copies, etc.

#ifndef REALM_RESILIENCY_H
#define REALM_RESILIENCY_H

#include <climits>
#include <vector>
#include <set>
#include <map>

#include "bytearray.h"
#include "processor.h"
#include "memory.h"
#include "instance.h"
#include "faults.h"

namespace Realm {

  // through the wonders of templates, users should never need to work with 
  //  these IDs directly
  enum ResiliencyID {
		RESILIENTID_RUN_UNTIL_COMPLETION,
    // as the name suggests, this should always be last, allowing apps/runtimes
    // sitting on top of Realm to use some of the ID space
    RESILIENTID_REALM_LAST = 10000,
  };

  namespace ResiliencyTechniques {
    struct RunUntilCompletion {
      static const ResiliencyID ID = RESILIENTID_RUN_UNTIL_COMPLETION;
      enum Result {
				COMPLETED_SUCCESSFULLY,
				COMPLETED_WITH_ERRORS,
      };
      Result result;
      int error_code;
      ByteArray error_details;
    };
  };

  class ResiliencyRequestTechnique {
  public:
    ResiliencyRequestTechnique();
    ~ResiliencyRequestTechnique(void);
    ResiliencyRequestTechnique& operator=(const ResiliencyRequestTechnique& rhs);

    template <typename T>
    ResiliencyRequestTechnique& add_resiliency(void);
    ResiliencyRequestTechnique& add_resiliency(ResiliencyID rid);

    ResiliencyRequestTechnique& add_technique(Processor response_proc, 
				  Processor::TaskFuncID response_task_id,
				  const void *payload = 0, size_t payload_size = 0);

		bool wants_technique(ResiliencyID technique);

    size_t request_count(void) const;
    bool empty(void) const;
    void clear(void);


  protected:
    //Processor proc;
    //Processor::TaskFuncID task_id;
    ByteArray user_data;
    std::set<ResiliencyID> requested_resilience_techniques;
//    template <typename S> friend bool serialize(S &s, const ResiliencyRequestTechnique &rrt);
//    template <typename S> friend bool deserialize(S &s, ResiliencyRequestTechnique &rrt);
  };

}; // namespace Realm

//#include "resiliency.inl"

#endif // ifdef REALM_RESILIENCY_H
